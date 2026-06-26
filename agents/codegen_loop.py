"""
Codegen Agent Loop
------------------
OpenCode generates code → Claude reviews → iterate until approved or max_iters.

OpenCode owns generation + execution. Claude owns judgment + review.
Feedback from Claude is injected into the next OpenCode prompt so it can fix its own output.

Usage:
    python -m xnch.agents.codegen_loop "implement pg_episodic_store.py with pgvector"
    python -m xnch.agents.codegen_loop --task-file task.md --workspace /tmp/codegen --max-iters 5
    python -m xnch.agents.codegen_loop --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# File types Claude will read when reviewing the workspace
_REVIEWABLE_EXTENSIONS = {
    ".py", ".ts", ".js", ".go", ".rs",
    ".yaml", ".yml", ".toml", ".json", ".sql", ".sh",
}

# Directories to skip when reading workspace
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".mypy_cache"}


@dataclass
class _Review:
    approved: bool
    feedback: str
    iteration: int


@dataclass
class CodegenResult:
    workspace: Path
    iterations: int
    approved: bool
    final_review: str


class CodegenLoop:
    """
    Drives an OpenCode → Claude review loop for a single coding task.

    OpenCode writes code to `workspace/`. Claude reads every file in `workspace/`
    and either approves or returns structured feedback. On NEEDS_CHANGES, the
    feedback becomes the next OpenCode prompt (continuing the same session so
    OpenCode has full context of what it already wrote).
    """

    def __init__(
        self,
        workspace: Path,
        max_iterations: int = 5,
        opencode_model: Optional[str] = None,
        claude_model: str = "claude-sonnet-4-6",
        verbose: bool = True,
    ) -> None:
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.opencode_model = opencode_model
        self.claude_model = claude_model
        self.verbose = verbose
        self._opencode_session: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> CodegenResult:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._log(f"Workspace: {self.workspace}")
        self._log(f"Task: {task[:120]}{'...' if len(task) > 120 else ''}")

        review: Optional[_Review] = None

        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\n{'─' * 60}")
            self._log(f"  ITERATION {iteration}/{self.max_iterations}")
            self._log(f"{'─' * 60}")

            # Step 1: OpenCode generates / fixes
            self._log("\n[opencode] Generating code…")
            self._run_opencode(self._build_codegen_prompt(task, review))

            # Step 2: Claude reviews
            self._log("\n[claude] Reviewing workspace…")
            review = self._run_claude_review(task, iteration)

            status = "APPROVED ✓" if review.approved else "NEEDS CHANGES ✗"
            self._log(f"\n[review] {status}")
            if not review.approved:
                preview = review.feedback[:400].replace("\n", " ")
                self._log(f"[feedback] {preview}…")

            if review.approved:
                self._log(f"\nDone in {iteration} iteration(s).")
                return CodegenResult(
                    workspace=self.workspace,
                    iterations=iteration,
                    approved=True,
                    final_review=review.feedback,
                )

        self._log(f"\nMax iterations ({self.max_iterations}) reached without approval.")
        return CodegenResult(
            workspace=self.workspace,
            iterations=self.max_iterations,
            approved=False,
            final_review=review.feedback if review else "",
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_codegen_prompt(self, task: str, previous_review: Optional[_Review]) -> str:
        if previous_review is None:
            return textwrap.dedent(f"""
                {task}

                Write the code to the workspace directory.
                Run it or test it if you can to verify it works before finishing.
            """).strip()

        return textwrap.dedent(f"""
            The previous implementation was reviewed and needs changes.

            ORIGINAL TASK:
            {task}

            REVIEWER FEEDBACK:
            {previous_review.feedback}

            Fix every issue listed above. Keep what was already correct.
            Re-run / re-test after fixing to confirm the issues are resolved.
        """).strip()

    # ------------------------------------------------------------------
    # OpenCode
    # ------------------------------------------------------------------

    def _run_opencode(self, prompt: str) -> None:
        cmd = [
            "opencode", "run",
            "--dir", str(self.workspace),
            "--dangerously-skip-permissions",
        ]

        if self.opencode_model:
            cmd.extend(["--model", self.opencode_model])

        if self._opencode_session:
            # Continue the same session so OpenCode has full context
            cmd.extend(["--session", self._opencode_session])

        cmd.append(prompt)

        result = subprocess.run(cmd, capture_output=False, timeout=600)

        if result.returncode != 0:
            raise RuntimeError(f"OpenCode exited with code {result.returncode}")

        # Capture the session ID on first run so we continue it next iteration
        if not self._opencode_session:
            self._opencode_session = self._latest_opencode_session()
            if self._opencode_session:
                self._log(f"[opencode] Session: {self._opencode_session}")

    def _latest_opencode_session(self) -> Optional[str]:
        """Parse most-recent session ID from `opencode session list`."""
        try:
            result = subprocess.run(
                ["opencode", "session", "list"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                stripped = line.strip()
                # Skip header and separator lines
                if not stripped or stripped.startswith("Session") or stripped.startswith("─"):
                    continue
                # First token on the line is the session ID
                token = stripped.split()[0]
                if token.startswith("ses_"):
                    return token
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Claude review
    # ------------------------------------------------------------------

    def _run_claude_review(self, task: str, iteration: int) -> _Review:
        files_block = self._gather_workspace_files()

        review_prompt = textwrap.dedent(f"""
            You are a senior engineer reviewing code generated for this task:

            TASK:
            {task}

            GENERATED FILES:
            {files_block}

            Review for:
            1. Correctness — does the code actually do what the task requires?
            2. Edge cases — are obvious failure modes handled?
            3. Security — no injection, no hardcoded secrets, safe defaults
            4. Code quality — clear naming, no dead code, no unnecessary complexity

            Respond with EXACTLY one of these two formats and nothing else:

            Format A (approved):
            APPROVED
            <one or two sentences explaining why it passes>

            Format B (needs changes):
            NEEDS_CHANGES
            - <specific actionable issue>
            - <specific actionable issue>
            ...

            Be decisive. Minor style preferences are NOT blockers.
            Only flag real correctness, security, or completeness problems.
        """).strip()

        result = subprocess.run(
            [
                "claude",
                "--print",
                "--model", self.claude_model,
                "--output-format", "text",
                review_prompt,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Claude review failed (exit {result.returncode}):\n{result.stderr}"
            )

        output = result.stdout.strip()
        approved = output.upper().startswith("APPROVED")

        return _Review(approved=approved, feedback=output, iteration=iteration)

    # ------------------------------------------------------------------
    # Workspace file reader
    # ------------------------------------------------------------------

    def _gather_workspace_files(self) -> str:
        parts: list[str] = []

        for path in sorted(self.workspace.rglob("*")):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue
            if path.suffix not in _REVIEWABLE_EXTENSIONS:
                continue

            relative = path.relative_to(self.workspace)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                parts.append(f"### {relative}\n```\n{content}\n```")
            except Exception:
                continue

        if not parts:
            return "(workspace is empty — no files generated yet)"

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Codegen agent loop: OpenCode generates → Claude reviews → iterate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python -m xnch.agents.codegen_loop "write a Python function that validates a JWT"
              python -m xnch.agents.codegen_loop --task-file task.md --max-iters 3
              python -m xnch.agents.codegen_loop "..." --workspace /tmp/codegen --claude-model claude-opus-4-8
        """),
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task description (inline). Use --task-file for longer tasks.",
    )
    parser.add_argument(
        "--task-file", "-f",
        metavar="FILE",
        help="Path to a file containing the task description.",
    )
    parser.add_argument(
        "--workspace", "-w",
        metavar="DIR",
        default="/tmp/codegen_workspace",
        help="Directory where OpenCode writes generated files (default: /tmp/codegen_workspace).",
    )
    parser.add_argument(
        "--max-iters", "-n",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of generate→review iterations (default: 5).",
    )
    parser.add_argument(
        "--opencode-model", "-om",
        metavar="PROVIDER/MODEL",
        default=None,
        help="Model for OpenCode (e.g. anthropic/claude-sonnet-4-6). Uses OpenCode default if omitted.",
    )
    parser.add_argument(
        "--claude-model", "-cm",
        metavar="MODEL",
        default="claude-sonnet-4-6",
        help="Claude model for review (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve task text
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    elif args.task:
        task = args.task.strip()
    else:
        print("Error: provide a task as an argument or via --task-file.", file=sys.stderr)
        sys.exit(1)

    loop = CodegenLoop(
        workspace=Path(args.workspace),
        max_iterations=args.max_iters,
        opencode_model=args.opencode_model,
        claude_model=args.claude_model,
        verbose=not args.quiet,
    )

    result = loop.run(task)

    # Exit code: 0 = approved, 1 = not approved within max_iters
    sys.exit(0 if result.approved else 1)


if __name__ == "__main__":
    main()
