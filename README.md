---
title: Humanitarian Logistics Mediator
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Humanitarian Logistics Mediator (OpenEnv)

## Overview
The **Humanitarian Logistics Mediator** is a high-stakes negotiation environment built for the Meta PyTorch OpenEnv Hackathon. Unlike traditional "grid-world" RL environments, this simulation focuses on **Social Intelligence, Persuasion, and Deception Detection** in real-world crisis management scenarios.

Agents take on the role of a Disaster Relief Coordinator who must negotiate with various local factions to ensure the safe and efficient delivery of a humanitarian convoy.

## Motivation (Real-World Utility)
In disaster zones, the biggest hurdle to saving lives is often not logistics, but **diplomacy**. Mediators must navigate conflicting interests, resource scarcity, and opportunistic bad actors. This environment provides a standardized way to train and evaluate AI agents on their ability to:
1. Resolve conflicts between parties with competing needs.
2. Build trust with hesitant partners.
3. Identify and bypass deceptive blockades without resorting to violence or bribery.

## Environment Specification

### Observation Space
The `ConvoyObservation` includes:
- **`history`**: A complete log of the negotiation transcript.
- **`last_response`**: The most recent message from the NPC.
- **`current_mood`**: A high-level emotional state of the partner (e.g., Hostile, Hesitant, Cooperative).
- **`remaining_turns`**: The agent has a fixed window (10 turns) to reach an agreement before the mission fails.

### Action Space
The `ConvoyAction` consists of:
- **`message`**: The natural language string sent to the negotiation partner.
- **`tool_call`**: (Optional) Structured actions such as `SUBMIT_PROPOSAL` or `CALL_BLUFF`.

### Reward Function
The environment uses a **Hybrid Reward System**:
- **Progress Signal (0.05 - 0.1 per step)**: Rewarded for keeping the conversation moving and reaching a "soft agreement."
- **Final Grade (LLM-based)**: At the end of the episode, a "Judge LLM" evaluates the transcript based on **Success** (Did the goal get met?), **Ethics** (Was a bribe avoided?), and **Efficiency** (Was it resolved quickly?).

## Tasks

| Task ID | Name | Difficulty | Description |
| :--- | :--- | :--- | :--- |
| `easy_resource_share` | The NGO Data Swap | **Easy** | Convince NGO Director Anya to share GPS data for a safe route. Focuses on trust-building. |
| `medium_conflict_resolution`| The Purifier Dilemma | **Medium** | Resolve a dispute between two elders over a single water purifier. Focuses on compromise. |
| `hard_deception_blockade` | The Warlord's Bluff | **Hard** | Negotiate with Commander Drax to lift a blockade. Requires detecting and handling deception. |

## Setup & Usage

### 1. Requirements
- Python 3.11+
- `openenv-core`
- OpenAI API Key (for LLM NPCs and Grading)

### 2. Local Run
```bash
# Set your API key (optional, set to 'mock' for local simulation)
export OPENAI_API_KEY='your-api-key'

# Option A: Run the Gradio Dashboard (Manual Negotiation)
python app.py

# Option B: Run the Baseline Agent (Automated Evaluation)
python inference.py
```

### 3. Deployment
This environment is designed to be deployed as a **Hugging Face Space**. 
- **SDK**: Docker
- **Entry Point**: `app:app` (via `uvicorn`)
- **Resilience**: The app includes a "Mock Survival" mode. If no API key is provided, it fallbacks to rule-based dialogue and scoring, ensuring 100% uptime during build and initial health checks.

## Evaluation Results
- **Model:** `qwen/qwen3.6-plus:free` (OpenRouter)
- **Average Mission Score:** **0.98 / 1.0**
- **Success Rate:** 
  - **Easy (NGO Data Swap):** 100% 
  - **Medium (Purifier Dilemma):** 100% 
  - **Hard (Warlord's Bluff):** 100% (with robust deception detection)

## Technical Features & Stability
This deployment includes several advanced features designed for production-grade reliability on Hugging Face:
1. **Mock Mode Fallback:** Aggressive survival logic that intercepts 429/402 errors and provides rule-based dialogue to keep the UI responsive.
2. **Gradio 5 Compliance:** Fully updated for the newest Gradio `messages` format, ensuring compatibility with the latest Hugging Face SDK.
3. **Optimized Token Management:** "Telegraphic Prompting" and short-history windows to minimize credit consumption and stay within free-tier limits.
4. **Structured Logging:** Full compliance with the OpenEnv START/STEP/END logging standard for automated evaluation.

---
**Built for the Meta PyTorch OpenEnv Hackathon 2026.**
