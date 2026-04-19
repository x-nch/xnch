"""GET /auth/public-key — serves xnch RS256 public key for token validation."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/public-key")
async def public_key(request: Request) -> dict:
    pem = request.app.state.keypair.public_pem.decode()
    return {"algorithm": "RS256", "public_key_pem": pem}
