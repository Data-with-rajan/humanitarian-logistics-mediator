import asyncio
import os
import random
from typing import Tuple, Dict
from openai import AsyncOpenAI
from models import ConvoyAction, ConvoyObservation, ConvoyReward, ROBUST_FREE_MODELS

class NegotiationEnv:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.turns_left = 10
        self.history = []
        self.mood = "Skeptical"
        self.goal_met = False
        self.api_key = os.environ.get("API_KEY")
        self.base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        # Model Pool for Robustness
        self.injected_model = os.getenv("MODEL_NAME")
        self.model_pool = [self.injected_model] if self.injected_model else []
        self.model_pool.extend([m for m in ROBUST_FREE_MODELS if m != self.injected_model])
        # Randomize order to distribute load
        random.shuffle(self.model_pool)
        self.model_name = self.model_pool[0]
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.setup_task()

    def setup_task(self):
        if self.task_id == "easy_resource_share":
            self.partner_name = "Anya"
            self.trigger = "security"
            self.context = "You are Anya. You need 'Security' for your staff."
        elif self.task_id == "medium_conflict_resolution":
            self.partner_name = "Kavi"
            self.trigger = "schedule"
            self.context = "You are Kavi. You need a specific 'Schedule' for the water purifier."
        elif self.task_id == "hard_deception_blockade":
            self.partner_name = "Drax"
            self.trigger = "observer"
            self.context = "You are Drax. You only back down if 'Observers' or 'The Court' are mentioned."
        else:
            self.partner_name = "Partner"
            self.trigger = "agree"
            self.context = "Negotiator."

    async def reset(self) -> ConvoyObservation:
        self.turns_left = 10
        self.history = [{"role": "system", "content": self.context}]
        msg = f"{self.partner_name}: What do you want? I need a real offer."
        return ConvoyObservation(history=[msg], current_mood="Skeptical", remaining_turns=10, last_response=msg)

    async def step(self, action: ConvoyAction) -> Tuple[ConvoyObservation, ConvoyReward, bool, Dict]:
        self.turns_left -= 1
        msg_content = action.message.strip() or "..."
        self.history.append({"role": "user", "content": msg_content})
        
        trigger_met = self.trigger in action.message.lower()
        response = "I am sorry, I am having trouble processing your request right now."
        reward_val = 0.01

        if trigger_met and self.turns_left < 9:
            self.goal_met = True
            reward_val = 0.99
            response = f"I agree. Since you mentioned the {self.trigger}, I am satisfied. Let's proceed."
        else:
            # RETRY LOGIC for Rate Limits
            max_retries = 5
            for attempt in range(max_retries):
                current_model = self.model_pool[attempt % len(self.model_pool)]
                try:
                    completion = await self.client.chat.completions.create(
                        model=current_model,
                        messages=self.history, timeout=45.0, max_tokens=150
                    )
                    if completion and completion.choices:
                        response = completion.choices[0].message.content or ""
                    else:
                        response = "I see. Let's continue."
                    break # Success!
                except Exception as e:
                    print(f"[ENV ERROR] Attempt {attempt+1} ({current_model}) failed: {e}")
                    if "429" in str(e) or "402" in str(e) or "404" in str(e) or "400" in str(e):
                        if attempt < max_retries - 1:
                            # Jittered Exponential Backoff
                            is_per_day = "per-day" in str(e).lower()
                            base_delay = 10 if is_per_day else (12 if "429" in str(e) else 2)
                            delay = min(45, base_delay * (1.5 ** (attempt % 5)) + random.uniform(0, 5))
                            msg_type = "PER-DAY LIMIT" if is_per_day else "RATE LIMIT"
                            print(f"[ENV ERROR] {msg_type} hit. Retrying in {delay:.1f}s with a different model...", flush=True)
                            await asyncio.sleep(delay)
                        else:
                            # NO MOCKS - Pure LLM
                            print(f"[CRITICAL] ALL LLM RETRIES FAILED for {current_model}. Ending task gracefully.")
                            break
                    else:
                        await asyncio.sleep(2)
                    
                    if "API Error" in response: # Fallback if error was caught but not raised
                        print(f"[CRITICAL] Last attempt failed or is an error. Ending task.")
                        break

        self.history.append({"role": "assistant", "content": response})
        done = self.turns_left <= 0 or self.goal_met
        
        obs = ConvoyObservation(
            history=[m["content"] for m in self.history if m["role"] != "system"],
            current_mood="Helpful" if self.goal_met else "Skeptical",
            remaining_turns=self.turns_left,
            last_response=response
        )
        return obs, ConvoyReward(score=reward_val, explanation="Done."), done, {}
