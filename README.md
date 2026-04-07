# OpsSim-AI: DevOps Incident Response Arena

*When systems begin to fail, every second becomes a decision.*

## Overview

OpsSim-AI is an OpenEnv-based DevOps incident response environment designed to evaluate how AI agents behave under pressure.

This project simulates realistic production failures where an agent must:
- interpret noisy signals
- diagnose system issues
- follow operational constraints
- make fast, high-stakes recovery decisions

Why it matters:
- Real outages are messy
- Logs are incomplete
- User complaints are ambiguous
- Wrong actions can deepen failure

OpsSim-AI brings that reality into a structured environment where agents are tested not on toy games, but on the kind of decisions that keep production systems alive.

## Problem Statement

Modern incident response is not just about detecting failures. It is about choosing the right action before the system deteriorates further.

Current benchmarking setups often fall short because they:
- simplify failures into static tasks
- ignore business constraints like SLA protection
- underrepresent uncertainty and operational trade-offs
- fail to model the cost of hesitation

In real DevOps environments:
- doing nothing is often an action with consequences
- saving one service may endanger another
- recovery is rarely a single-step fix
- delays translate into user pain, revenue loss, and cascading instability

We built OpsSim-AI to capture that urgency.

## Solution

OpsSim-AI solves this by combining:
- a structured OpenEnv environment
- typed observations, actions, and rewards
- scenario-driven incident simulations
- an LLM-based decision layer for action selection

The system presents the agent with evolving production states across three difficulty levels:
- `easy`: direct configuration recovery
- `medium`: hidden bug diagnosis from indirect signals
- `hard`: SLA-aware incident mitigation under pressure

Using OpenEnv, each task follows a consistent interaction loop:
- reset environment
- observe state
- choose action
- receive reward feedback
- continue until resolved or terminated

The AI agent does not simply classify a state. It must reason through consequences, adapt across steps, and recover systems while minimizing operational damage.

## System Architecture

### `env.py`
The simulation engine.

Responsibilities:
- loads task scenarios
- maintains environment state
- applies transitions
- computes rewards
- enforces penalties, guardrails, and SLA outcomes

This is the operational heart of the system.

### `inference.py`
The LLM decision layer and evaluator.

Responsibilities:
- reads environment observations
- prompts the model for next actions
- executes environment steps
- prints standardized evaluation traces
- computes normalized task scores

This is where the AI agent turns observation into action.

### `models.py`
The typed contract.

Responsibilities:
- defines `Observation`
- defines `Action`
- defines `Reward`

Using Pydantic ensures clean structure, consistent interfaces, and OpenEnv-aligned interaction semantics.

## Reward Function Design

The reward design is built to feel operationally real, not artificially convenient.

### Core Philosophy

In real incident response:
- delay is expensive
- useless actions are expensive
- risky actions can be catastrophic
- partial improvement matters
- violating SLA rules can end the episode immediately

This reward system captures that tension.

### Dynamic SLA Bleed

The environment applies **dynamic bleed**, which represents ongoing system damage while the incident remains unresolved.

Intuition:
- if the system is degraded, the system keeps bleeding
- the longer critical conditions persist, the more costly each step becomes
- stability must be restored, not merely observed

This makes the environment feel alive. The system does not wait politely while the agent thinks.

### Why Doing Nothing Is Costly

`do_nothing` is explicitly penalized.

Why:
- in real outages, inaction is often a decision
- waiting while systems degrade can be as dangerous as a bad intervention
- incident response rewards decisive, informed action

This encourages the agent to engage with the problem rather than stall.

### Trade-Offs Matter

The hardest task is built around operational trade-offs.

Sometimes the agent must choose between:
- protecting critical services
- reducing overall system pressure
- avoiding harmful interventions
- respecting SLA and playbook guardrails

That means the reward is not about blindly maximizing one metric. It is about making the least dangerous decision in a failing system.

### Reward Components

The hard environment combines several signals:

- **Dynamic bleed**
  State-based penalties that reflect ongoing system degradation

- **Action penalties**
  Punish invalid, repeated, destructive, or wasteful actions

- **Urgency penalty**
  Adds pressure over time so late recovery is less valuable than fast recovery

- **Progress rewards**
  Rewards meaningful improvements, including partial stabilization and SLA improvement

- **SLA guardrails**
  Severe failures or forbidden actions can terminate the episode with strong penalties

### Why This Design Is Powerful

This design is realistic because it mirrors how production incidents actually unfold:
- damage accumulates
- time matters
- careless actions hurt
- partial recovery has value
- business constraints shape technical decisions

The result is a reward function that does more than score correctness. It evaluates operational judgment.

## Features

- Deterministic execution for scenario sequencing and grading
- SLA-aware decision environments
- Multi-step reasoning across escalating incident complexity
- Real-world DevOps and outage-inspired scenarios
- Typed Pydantic observation, action, and reward models
- OpenEnv-compatible task structure
- Standardized evaluation traces for reproducibility
- Scalable dataset design using scenario JSON files
- Reward shaping that reflects operational urgency and trade-offs

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd OpsSim-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

#### Linux / macOS
```bash
source .venv/bin/activate
```

#### Windows PowerShell
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up `.env`

Create a `.env` file in the project root and add:

```env
HF_TOKEN=your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

### 5. Run the Project

```bash
python inference.py
```

## Example Output

```text
[START] task=hard_scenario_1 env=ops-sim model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=restart(database) reward=-0.25 done=false error=null
[STEP] step=2 action=reroute_traffic reward=0.30 done=false error=null
[STEP] step=3 action=scale_backend reward=1.10 done=true error=null
[END] success=true steps=3 rewards=-0.25,0.30,1.10
```

## Hackathon Compliance

This project is designed to align with the hackathon requirements:

- OpenEnv-style environment interface
- Typed `Observation`, `Action`, and `Reward` models using Pydantic
- Three structured tasks: `easy -> medium -> hard`
- Programmatic scoring normalized to the `0.0 - 1.0` range
- Deterministic environment behavior for task sequencing and grading
- OpenAI client used for all LLM calls
- Standardized inference output format for evaluation

## Future Improvements

- Add richer multi-service dependency graphs for more complex outage propagation
- Expand incident datasets with domain-specific failure modes
- Introduce postmortem generation for explainable recovery analysis
- Add human-vs-agent benchmarking for operational comparison

## Conclusion

OpsSim-AI is built around a simple belief: reliability is tested when systems are already under stress.

This project does not ask an agent to solve a puzzle. It asks the agent to respond when infrastructure is degrading, signals are noisy, and hesitation has a cost.

That is where dependable AI begins to matter.
