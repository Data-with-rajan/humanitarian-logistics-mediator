from typing import List, Optional
from pydantic import BaseModel, Field

class ConvoyAction(BaseModel):
    message: str = Field(..., description="The message sent to the partner.")

class ConvoyObservation(BaseModel):
    history: List[str]
    current_mood: str
    remaining_turns: int
    last_response: str

class ConvoyReward(BaseModel):
    score: float
    explanation: str
