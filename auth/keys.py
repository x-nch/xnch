"""RS256 key pair management. Generates and persists a 2048-bit RSA key pair."""
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


@dataclass(frozen=True)
class KeyPair:
    private_pem: bytes
    public_pem: bytes


def load_or_generate_keypair(keys_dir: Path) -> KeyPair:
    keys_dir.mkdir(parents=True, exist_ok=True)
    private_path = keys_dir / "private.pem"
    public_path = keys_dir / "public.pem"

    if private_path.exists() and public_path.exists():
        return KeyPair(
            private_pem=private_path.read_bytes(),
            public_pem=public_path.read_bytes(),
        )

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path.write_bytes(private_pem)
    private_path.chmod(0o600)
    public_path.write_bytes(public_pem)
    return KeyPair(private_pem=private_pem, public_pem=public_pem)
