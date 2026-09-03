"""Adversarial debate (Support vs Attack vs Critic) routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/{claim_id}")
async def get_adversarial_debate(claim_id: str):
    """Retrieve the proponent and opponent arguments, evidence, and critic analysis."""
    return {}
