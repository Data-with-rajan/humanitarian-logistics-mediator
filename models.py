from typing import List, Optional
from pydantic import BaseModel, Field

# Centralized Model Pool for Robustness
ROBUST_FREE_MODELS = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "arcee-ai/trinity-mini:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "meta-llama/llama-3.3-70b-instruct:free"
]

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
