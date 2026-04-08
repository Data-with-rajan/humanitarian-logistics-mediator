import asyncio, os, subprocess, traceback, sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict

# Add parent directory to sys.path for root module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import NegotiationEnv
from models import ConvoyAction
import gradio as gr

# --- 1. STARTUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs the background inference required by the grader
    print("[SERVER] Starting baseline inference in background...", flush=True)
    try:
        # Path to inference.py is now in parent directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        inference_path = os.path.join(root_dir, "inference.py")
        subprocess.Popen([sys.executable, inference_path])
    except Exception as e:
        print(f"[SERVER ERROR] {e}", flush=True)
    yield

# Standard FastAPI instance
app = FastAPI(lifespan=lifespan)
envs: Dict[str, NegotiationEnv] = {}

# --- 2. API ENDPOINTS (REQUIRED BY OPENENV GRADER) ---
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json() if await request.body() else {}
        tid = data.get("task_id", "easy_resource_share")
        envs[tid] = NegotiationEnv(tid)
        obs = await envs[tid].reset()
        return JSONResponse(obs.model_dump())
    except Exception:
        tid = "easy_resource_share"
        envs[tid] = NegotiationEnv(tid)
        obs = await envs[tid].reset()
        return JSONResponse(obs.model_dump())

@app.post("/step")
async def step(request: Request):
    try:
        data = await request.json() if await request.body() else {}
        tid = data.get("task_id", "easy_resource_share")
        act = data.get("action", {})
        if tid not in envs: envs[tid] = NegotiationEnv(tid)
        action_obj = ConvoyAction(message=act.get("message", ""))
        obs, reward, done, info = await envs[tid].step(action_obj)
        return JSONResponse({
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": bool(done)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# --- 3. UI LOGIC ---
async def process_step(message, history, action_log, task_id):
    try:
        if history is None: history = []
        if not history or task_id not in envs:
            envs[task_id] = NegotiationEnv(task_id)
            obs = await envs[task_id].reset()
            history = [{"role": "assistant", "content": obs.last_response}]
            return history, "Skeptical", 10, 0.0, "Negotiation Started."
        
        obs, reward, done, info = await envs[task_id].step(ConvoyAction(message=message))
        res = obs.last_response
        if done: res += f"\n\n[DONE - Final Reward: {reward.score}]"
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": res})
        new_log = f"Step {11-obs.remaining_turns}: Reward {reward.score}\n" + action_log
        return history, obs.current_mood, obs.remaining_turns, reward.score, new_log
    except Exception as e:
        if history is None: history = []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, "Error", 0, 0.0, f"UI Error: {e}"

# --- 4. BUILD DASHBOARD ---
with gr.Blocks(title="🌍 OpenEnv Mediator") as demo:
    gr.Markdown("# 🌍 Humanitarian Logistics Mediator\n**Real-Time RL Dashboard**")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📡 Observation Space")
            task_select = gr.Dropdown(["easy_resource_share", "medium_conflict_resolution", "hard_deception_blockade"], value="easy_resource_share", label="Select Mission")
            mood_disp = gr.Textbox(label="NPC Mood", value="Neutral", interactive=False)
            turns_disp = gr.Number(label="Turns Left", value=10, interactive=False)
            reward_disp = gr.Number(label="Step Reward", value=0.0, interactive=False)
            reset_btn = gr.Button("Reset Episode", variant="stop")
            action_log_disp = gr.Textbox(label="Action History", lines=10, value="Waiting for first step...")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Communication Channel")
            chatbot = gr.Chatbot(label="Negotiation Log")
            msg_input = gr.Textbox(label="Your Message", placeholder="Propose your deal...")
            submit_btn = gr.Button("Submit Action", variant="primary")

    submit_btn.click(
        process_step, 
        inputs=[msg_input, chatbot, action_log_disp, task_select], 
        outputs=[chatbot, mood_disp, turns_disp, reward_disp, action_log_disp]
    )
    reset_btn.click(
        lambda: ([], "Skeptical", 10, 0.0, "Resetting..."), 
        outputs=[chatbot, mood_disp, turns_disp, reward_disp, action_log_disp]
    )

app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
