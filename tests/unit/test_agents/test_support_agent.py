from mascv.models.argument import Argument


def test_argument_schema():

    argument = Argument(
        agent_name="SupportAgent",
        claim_id="C1",
        stance="FOR",
        premises=[
            "The proposed method improved accuracy."
        ],
        cited_evidence_ids=[
            "E-001"
        ],
        conclusion=(
            "The available evidence provides "
            "support for the claim."
        ),
        strength="Moderate",
        identified_limitations=[
            "Only one benchmark was evaluated."
        ],
    )

    assert argument.agent_name == "SupportAgent"

    assert argument.claim_id == "C1"

    assert "E-001" in (
        argument.cited_evidence_ids
    )

    assert argument.strength in {
        "Strong",
        "Moderate",
        "Weak",
    }