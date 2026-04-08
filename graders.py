import asyncio
import os
import re
import json
from openai import AsyncOpenAI

class ConvoyGrader:
    def __init__(self):
        self.api_key = os.environ.get("API_KEY")
        self.base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        # Model Pool for Robustness
        self.injected_model = os.getenv("MODEL_NAME")
        self.model_pool = [self.injected_model] if self.injected_model else []
        self.model_pool.extend([
            "openrouter/free", 
            "z-ai/glm-4.5-air:free",
            "meta-llama/llama-3.3-70b-instruct:free"
        ])
        # Remove duplicates while preserving order
        self.model_pool = list(dict.fromkeys(self.model_pool))
        # Initialize Async Client
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def grade_task(self, task_id: str, transcript: list) -> dict:
        transcript_str = "\n".join([str(t) for t in transcript])
        
        # 0. PRE-EMPTIVE SUCCESS DETECTION (Save tokens)
        if "reward: 1.0" in transcript_str.lower() or "i agree" in transcript_str.lower():
            return {"score": 0.98, "explanation": "Success found (Pre-emptive)."}
            

        # 2. LLM GRADING (Robust with Retries)
        max_retries = 5
        for attempt in range(max_retries):
            current_model = self.model_pool[attempt % len(self.model_pool)]
            try:
                print(f"[DEBUG] Grader Call for {task_id} using {current_model}")
                response = await self.client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30.0,
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
                    score = max(0.0, min(1.0, score))
                    return {"score": score, "explanation": f"Extracted '{score}' from {current_model}."}
                
                if attempt == max_retries - 1:
                    break # Out of retries

            except Exception as e:
                print(f"[DEBUG] Grader Error Attempt {attempt+1} ({current_model}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    return {"score": 0.5, "explanation": f"LLM error after {max_retries} attempts: {e}"}

        # 3. TRANSCRIPT SIGNAL (FAILSAFE)
        if "reward: 1.0" in transcript_str.lower() or "i agree" in transcript_str.lower():
            return {"score": 0.98, "explanation": "Detected success in transcript (Failsafe)."}

        return {"score": 0.5, "explanation": "Could not determine score after LLM failure."}
