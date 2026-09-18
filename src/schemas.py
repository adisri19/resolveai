"""Request/response models. The Decision model doubles as the Gemini response schema."""
from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field

# The complete action vocabulary. Anything outside this list is a hallucination
# and is rejected before it reaches the database.
ACTIONS = (
    "APPROVE_REFUND_OR_REPLACEMENT",
    "REQUEST_PHOTOS",
    "APPROVE_RETURN",
    "REJECT_OPENED_ITEM",
    "REJECT_FOOD_RETURN",
    "REJECT_OUTSIDE_WINDOW",
    "REPLACE_CORRECT_ITEM",
    "APPROVE_REPLACEMENT",
    "REQUEST_DEFECT_EVIDENCE",
    "WAIT_AND_TRACK",
    "OPEN_SHIPPING_INVESTIGATION",
    "OFFER_REPLACEMENT_OR_REFUND",
    "CANCEL_AND_REFUND",
    "CANNOT_CANCEL_AFTER_DISPATCH",
    "NEEDS_MORE_INFORMATION",
)

Action = Literal[ACTIONS]  # type: ignore[valid-type]


class Credentials(BaseModel):
    email: EmailStr
    # bcrypt silently ignores bytes past 72; cap it here rather than let two
    # different long passwords authenticate the same account.
    password: str = Field(min_length=8, max_length=72)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class User(BaseModel):
    id: int
    email: str
    created_at: str


class TicketIn(BaseModel):
    """The message is the only required field; the rest are structured facts the
    policies need. Absent == unknown, and unknown is what drives NEEDS_MORE_INFORMATION."""
    message: str = Field(min_length=1, max_length=4000)
    order_value_inr: Optional[float] = Field(default=None, ge=0)
    days_since_delivery: Optional[int] = Field(default=None, ge=0)
    days_since_dispatch: Optional[int] = Field(default=None, ge=0)
    product_type: Optional[Literal["food", "non_food", "mixed", "unknown"]] = None
    opened_status: Optional[Literal["opened", "unopened", "unknown"]] = None
    order_status: Optional[Literal["processing", "dispatched", "delivered", "unknown"]] = None


class Decision(BaseModel):
    """Structured LLM output. Passed to Gemini as response_schema, then re-validated."""
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    sources: List[str]


class TicketOut(BaseModel):
    id: int
    message: str
    created_at: str
    decision: Optional[Decision] = None
