"""Turn a ticket + retrieved policy rules into a validated, structured decision."""
import os
from typing import List, Tuple

from .retrieval import Chunk, retrieve
from .schemas import ACTIONS, Decision, TicketIn

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# What each action *does*. Deliberately no eligibility conditions here -- those
# come from the retrieved policy rules, which is the point of doing retrieval.
ACTION_GLOSSARY = """
APPROVE_REFUND_OR_REPLACEMENT - grant a refund or replacement straight away
REQUEST_PHOTOS                - ask for photographs before approving anything
APPROVE_RETURN                - accept a change-of-mind return
REJECT_OPENED_ITEM            - refuse a change-of-mind return because the item was opened
REJECT_FOOD_RETURN            - refuse a change-of-mind return because the item is food
REJECT_OUTSIDE_WINDOW         - refuse because the report falls outside the policy's time window
REPLACE_CORRECT_ITEM          - send the item the customer actually ordered
APPROVE_REPLACEMENT           - approve a replacement for a functional defect
REQUEST_DEFECT_EVIDENCE       - ask for evidence of the defect before approving a replacement
WAIT_AND_TRACK                - tell the customer to keep waiting and tracking
OPEN_SHIPPING_INVESTIGATION   - raise an investigation with the carrier
OFFER_REPLACEMENT_OR_REFUND   - offer a replacement or refund for a lost shipment
CANCEL_AND_REFUND             - cancel the order and refund it
CANNOT_CANCEL_AFTER_DISPATCH  - tell the customer cancellation is no longer possible
NEEDS_MORE_INFORMATION        - a fact the policies require is missing; ask for it
""".strip()

SYSTEM_RULES = f"""You are a support-ticket decision engine for an Indian e-commerce retailer.

Decide the single correct action using ONLY the POLICY RULES supplied below. Do not
use outside knowledge and do not invent policy. Every policy rule is quoted verbatim
from the company's knowledge base and is prefixed with the file it came from.

Available actions:
{ACTION_GLOSSARY}

Rules of engagement:
- Choose exactly one action from the list above.
- If a fact a policy requires (delivery date, dispatch date, order value, product type,
  opened/unopened status) is missing or "unknown" AND that fact changes the outcome,
  answer NEEDS_MORE_INFORMATION. Never guess the missing fact.
- If no supplied policy rule covers the situation, answer NEEDS_MORE_INFORMATION.
- Check time windows and value thresholds arithmetically against the ticket facts.
- "reason" must be one or two sentences naming the rule and the facts you applied.
- "sources" must list only the filenames of policy rules you actually relied on.
- "confidence" is 0.0-1.0: high when one rule clearly matches, low when the facts are thin."""


def _format_facts(ticket: TicketIn) -> str:
    def show(value):
        return "unknown" if value is None or value == "unknown" else value

    return "\n".join(
        [
            f'Customer message: "{ticket.message}"',
            f"Order value (INR): {show(ticket.order_value_inr)}",
            f"Days since delivery: {show(ticket.days_since_delivery)}",
            f"Days since dispatch: {show(ticket.days_since_dispatch)}",
            f"Product type: {show(ticket.product_type)}",
            f"Opened status: {show(ticket.opened_status)}",
            f"Order status: {show(ticket.order_status)}",
        ]
    )


def build_prompt(ticket: TicketIn, chunks: List[Chunk]) -> str:
    context = "\n".join(f"[{c.source}] {c.text}" for c in chunks)
    return f"{SYSTEM_RULES}\n\nPOLICY RULES:\n{context}\n\nTICKET FACTS:\n{_format_facts(ticket)}"


def _generate(prompt: str) -> Decision:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Decision,   # Gemini constrains generation to this schema
            temperature=0,              # decisions should be reproducible
        ),
    )
    if resp.parsed is None:
        raise ValueError(f"Gemini returned unparseable output: {resp.text!r}")
    return resp.parsed


class DecisionUnavailable(RuntimeError):
    """The model could not be reached, or would not produce usable output.

    Deliberately NOT a NEEDS_MORE_INFORMATION decision. That action means the model
    answered "these facts are too thin to decide"; this means it did not answer at
    all. Collapsing the two would persist a fabricated decision and make an outage
    or a quota limit read as poor accuracy.
    """

    def __init__(self, cause: BaseException, chunks: List[Chunk]):
        super().__init__(str(cause))
        self.chunks = chunks


def decide(ticket: TicketIn) -> Tuple[Decision, List[Chunk]]:
    """Retrieve, generate, validate.

    Raises DecisionUnavailable if the model never produced a usable answer. Callers
    decide what to do about that; none of them invent a decision to fill the gap."""
    try:
        chunks = retrieve(ticket.message + "\n" + _format_facts(ticket))
    except RuntimeError:
        # Missing key or empty knowledge base: a misconfiguration with an actionable
        # message, not a transient failure. Let it through unchanged.
        raise
    except Exception as exc:  # noqa: BLE001 - the embedding API is as fallible as the generation one
        raise DecisionUnavailable(exc, [])

    prompt = build_prompt(ticket, chunks)

    last_error = None
    for _ in range(2):  # one retry; the API is occasionally flaky, not usually wrong
        try:
            return _ground(_generate(prompt), chunks), chunks
        except Exception as exc:  # noqa: BLE001 - transport, quota and schema failures are all "no answer"
            last_error = exc
    raise DecisionUnavailable(last_error, chunks)


def _ground(decision: Decision, chunks: List[Chunk]) -> Decision:
    """Keep only citations that point at policy rules we actually retrieved."""
    retrieved = {c.source for c in chunks}
    sources = [s for s in dict.fromkeys(decision.sources) if s in retrieved]
    if not sources and decision.action != "NEEDS_MORE_INFORMATION":
        # The model decided but cited nothing real; fall back to the top hit so the
        # stored decision is still traceable to a document.
        sources = [chunks[0].source] if chunks else []
    return decision.model_copy(update={"sources": sources})


assert set(ACTIONS) == {line.split()[0] for line in ACTION_GLOSSARY.splitlines()}, (
    "ACTION_GLOSSARY and schemas.ACTIONS have drifted apart"
)
