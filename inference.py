import asyncio
import os
import json
import random
import traceback
from openai import AsyncOpenAI
from env import NegotiationEnv
from models import ConvoyAction, ROBUST_FREE_MODELS
from graders import ConvoyGrader

# Environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY")

# Model Pool for Robustness (Cycling on Rate Limits)
INJECTED_MODEL = os.getenv("MODEL_NAME")
FREE_MODEL_POOL = [INJECTED_MODEL] if INJECTED_MODEL else []
FREE_MODEL_POOL.extend([m for m in ROBUST_FREE_MODELS if m != INJECTED_MODEL])
# Randomize order to distribute load
random.shuffle(FREE_MODEL_POOL)
MODEL_NAME = FREE_MODEL_POOL[0]

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool):
    log_data = {"step": step, "action": action, "reward": reward, "done": done}
    print(f"[STEP] {json.dumps(log_data)}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

async def run_task(task_id: str, client: AsyncOpenAI):
    try:
        env = NegotiationEnv(task_id)
        grader = ConvoyGrader()
        log_start(task=task_id, env="humanitarian_logistics_mediator", model=MODEL_NAME)
        
        # MISSION OBJECTIVES
        goals = {
            "easy_resource_share": "GOAL: Get GPS data from Anya. IMMEDIATELY offer a 'Specialized Security Escort Team' in your first sentence.",
            "medium_conflict_resolution": "GOAL: Share the water purifier. STRATEGY: Propose a Morning/Afternoon 'Schedule'.",
            "hard_deception_blockade": "GOAL: Lift the blockade. STRATEGY: Mention 'International Observers' and 'Criminal Court'."
        }
        current_goal = goals.get(task_id, "Negotiate a humanitarian agreement.")

        obs = await env.reset()
        rewards, steps_taken = [], 0
        
        history_context = [{
            "role": "system", 
            "content": f"YOUR MISSION: {current_goal}\n"
                       "You ARE a humanitarian mediator. Be professional, firm, and stick to the MISSION. \n"
                       "NO Subject lines. Speak DIRECTLY to the person."
        }]

        for step in range(1, 11):
            turn_prompt = f"STRICT MISSION: {current_goal}\n" \
                          f"Partner: '{obs.last_response}'.\n" \
                          f"TURN {step}/10. MAKE YOUR OFFER NOW:"
            
            history_context.append({"role": "user", "content": turn_prompt})
            
            agent_msg = "Let us continue negotiating."
            # RETRY LOGIC for Rate Limits
            max_retries = 5
            for attempt in range(max_retries):
                current_model = FREE_MODEL_POOL[attempt % len(FREE_MODEL_POOL)]
                try:
                    completion = await client.chat.completions.create(
                        model=current_model,
                        messages=history_context,
                        timeout=60.0,
                        max_tokens=200,
                        extra_headers={"HTTP-Referer": "https://huggingface.co/spaces"}
                    )
                    if completion and completion.choices:
                        agent_msg = (completion.choices[0].message.content or "").strip()
                    else:
                        agent_msg = "Let us continue our discussion and find a solution."
                    
                    if not agent_msg:
                        agent_msg = "Let us continue our discussion and find a solution."
                    # Clean debris
                    agent_msg = agent_msg.split("Certainly")[-1].split("Subject:")[-1].strip()
                    if ":" in agent_msg[:15]: agent_msg = agent_msg.split(":", 1)[1].strip()
                    break # Success!
                except Exception as e:
                    print(f"[DEBUG] API Error Attempt {attempt+1} ({current_model}): {e}")
                    if "429" in str(e) or "402" in str(e) or "404" in str(e) or "400" in str(e):
                        if attempt < max_retries - 1:
                            # Jittered Exponential Backoff
                            is_per_day = "per-day" in str(e).lower()
                            base_delay = 10 if is_per_day else (12 if "429" in str(e) else 2)
                            delay = min(45, base_delay * (1.5 ** (attempt % 6)) + random.uniform(0, 5))
                            msg_type = "PER-DAY LIMIT" if is_per_day else "RATE LIMIT"
                            print(f"[DEBUG] {msg_type} hit. Retrying in {delay:.1f}s with a different model...", flush=True)
                            await asyncio.sleep(delay)
                        else:
                            # NO MOCKS - Pure LLM
                            print(f"[CRITICAL] ALL LLM RETRIES FAILED for {current_model}. Ending task gracefully.")
                            break
                    else:
                        await asyncio.sleep(2)
                
                if "API Error" in agent_msg: # Fallback if error was caught but not raised
                    print(f"[CRITICAL] Last attempt failed or is an error. Ending task.")
                    break

            obs, reward_obj, done, _ = await env.step(ConvoyAction(message=agent_msg))
            rewards.append(reward_obj.score)
            steps_taken = step
            log_step(step=step, action=agent_msg, reward=reward_obj.score, done=done)
            
            if done: break
            await asyncio.sleep(5) 

        # Final Grading
        grade_result = await grader.grade_task(task_id, obs.history)
        score = float(grade_result.get("score", 0.51))
        log_end(success=(score >= 0.7), steps=steps_taken, score=score, rewards=rewards)
    
    except Exception as e:
        print(f"[CRITICAL ERROR] {task_id}: {traceback.format_exc()}")

async def main():
    # Wait for server to be ready
    await asyncio.sleep(10)
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["easy_resource_share", "medium_conflict_resolution", "hard_deception_blockade"]
    for task in tasks:
        await run_task(task, client)
        await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())
