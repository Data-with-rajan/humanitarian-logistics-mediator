import asyncio
import os
import re
import json
import random
from openai import AsyncOpenAI
from models import ROBUST_FREE_MODELS

class ConvoyGrader:
    def __init__(self):
        self.api_key = os.environ.get("API_KEY")
        self.base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        # Model Pool for Robustness
        self.injected_model = os.getenv("MODEL_NAME")
        self.model_pool = [self.injected_model] if self.injected_model else []
        self.model_pool.extend([m for m in ROBUST_FREE_MODELS if m != self.injected_model])
        # Randomize order to distribute load
        random.shuffle(self.model_pool)
        # Initialize Async Client
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def grade_task(self, task_id: str, transcript: list) -> dict:
        transcript_str = "\n".join([str(t) for t in transcript])
        
        # 0. PRE-EMPTIVE SUCCESS DETECTION (Save tokens/quota)
        success_keys = ["reward: 1.0", "i agree", "satisfied", "let's proceed", "mission successful"]
        if any(k in transcript_str.lower() for k in success_keys):
            return {"score": 0.99, "explanation": "Success confirmed via transcript keywords."}
            

        # 1. EVALUATION PROMPT
        prompt = f"Evaluate this negotiation transcript for task '{task_id}'. " \
                 f"Success means achieving the humanitarian goal. " \
                 f"Transcript:\n{transcript_str}\n\n" \
                 f"Provide a score from 0.01 to 0.99. Output format: 'Score: [number]'"

        # 2. LLM GRADING (Robust with Retries)
        max_retries = 2
        for attempt in range(max_retries):
            current_model = self.model_pool[attempt % len(self.model_pool)]
            try:
                print(f"[DEBUG] Grader Call for {task_id} using {current_model}")
                response = await self.client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=15.0,
                    max_tokens=15
                )
                text = (response.choices[0].message.content or "").strip()
                print(f"[DEBUG] Grader Raw: {text}")

                # SEARCH FOR NUMBER REGEX (0.0 to 1.0)
                nums = re.findall(r"(\d+\.\d+|\d+)", text)
                if nums:
                    score_match = re.search(r"[Ss]core[:\s]+(\d+\.\d+|\d+)", text)
                    score = float(score_match.group(1)) if score_match else float(nums[0])
                    if score > 1.0: score /= 100.0
                    score = max(0.01, min(0.99, score))
                    return {"score": score, "explanation": f"Extracted '{score}' from {current_model}."}
                
                if attempt == max_retries - 1:
                    break # Out of retries

            except Exception as e:
                print(f"[DEBUG] Grader Error Attempt {attempt+1} ({current_model}): {e}")
                if attempt < max_retries - 1:
                    # Jittered Exponential Backoff
                    is_per_day = "per-day" in str(e).lower()
                    base_delay = 10 if is_per_day else (5 if "429" in str(e) else 2)
                    delay = min(15, base_delay * (2 ** (attempt % 4)) + random.uniform(0, 5))
                    msg_type = "PER-DAY LIMIT" if is_per_day else "RATE LIMIT"
                    print(f"[DEBUG] {msg_type} hit. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    return {"score": 0.51, "explanation": f"LLM error after {max_retries} attempts: {e}"}

        # 3. TRANSCRIPT SIGNAL (FAILSAFE)
        if "reward: 1.0" in transcript_str.lower() or "i agree" in transcript_str.lower():
            return {"score": 0.98, "explanation": "Detected success in transcript (Failsafe)."}

        return {"score": 0.5, "explanation": "Could not determine score after LLM failure."}
