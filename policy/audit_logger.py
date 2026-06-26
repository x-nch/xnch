import logging
import datetime
from pathlib import Path

class PolicyAuditLogger:
    def __init__(self, log_file: str = "policy_audit.log"):
        self.logger = logging.getLogger("PolicyAuditLogger")
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def log_verdict(self, action: str, actor: str, verdict: str, rule_id: str):
        message = f"Action: {action} | Actor: {actor} | Verdict: {verdict} | Rule: {rule_id}"
        self.logger.info(message)