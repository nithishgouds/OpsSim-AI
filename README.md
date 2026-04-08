---
title: OpsSim-AI
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

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

### Easy Task Reward Design

The `easy` task is intentionally simple in structure, but not trivial in behavior. Its reward design is built around the idea of **fast and correct configuration recovery**.

At this level, the environment rewards the agent for identifying the single action that truly resolves the issue. The goal is not to blindly explore every option. The goal is to read the user message, connect it to the logs, and act with precision.

The reward behaves like a lightweight operational signal:
- every step carries a small cost, because even simple incidents become more expensive when they drag on
- invalid actions are penalized, because random guessing is not real diagnosis
- the correct action receives a strong positive reward and ends the episode
- misleading but plausible actions can still produce useful learning signal through red-herring handling
- repeating a known bad path becomes more costly than the first mistake

In plain terms, the `easy` reward function teaches the agent one important habit: **do not confuse motion with progress**.

Why this matters:
- many real incidents begin with something that looks obvious
- engineers often lose time following the wrong symptom
- the best response is not more action, but the right action

So even though the task is easier, the reward design still captures a realistic principle: speed matters, but precision matters more.

### Medium Task Reward Design

The `medium` task introduces a more subtle reality: systems may appear healthy while users are still failing.

Here, the reward function is designed around **structured diagnosis under uncertainty**.

The agent is no longer solving a single visible configuration problem. Instead, it must infer a hidden bug from indirect evidence such as user complaints, shifting hints, and the effect of earlier actions. That means the reward design must encourage disciplined reasoning across multiple steps.

The medium reward function does this by combining:
- a step-based cost that discourages wandering
- penalties for invalid actions, because careless action selection still has operational cost
- repeat penalties, because looping on the same action is one of the most common forms of failed diagnosis
- action-specific rewards from transition rules, so meaningful investigative behavior is recognized
- completion reward through the successful satisfaction of the scenario's success condition

What makes this powerful is that the reward does not only celebrate the final fix. It also recognizes **useful diagnostic behavior** along the way.

That reflects real-world engineering:
- good incident response is often a sequence of narrowing possibilities
- useful analysis steps may not solve the outage immediately, but they reduce uncertainty
- repeating the same ineffective move is operationally expensive and psychologically realistic

The `medium` task reward design therefore teaches the agent to behave like a thoughtful operator:
- investigate
- infer
- adapt
- then fix

### Hard Task Reward Design

The `hard` task is where the reward function becomes fully operational in spirit.

This task is designed around **incident management under pressure**, where every action has consequences and some choices may stabilize one part of the system while harming another. The reward function is not merely scoring correctness. It is measuring whether the agent behaves like a responsible incident commander.

At this level, reward emerges from several interacting forces.

#### Dynamic Bleed

Dynamic bleed represents the cost of leaving the system in a degraded state.

This is one of the most important ideas in the environment:
- the incident is not frozen in time
- unresolved failures continue to hurt
- every step taken in a bad state increases operational damage

This mirrors real outages, where degraded routing, overloaded services, or broken dependencies continue causing user pain until they are actively addressed.

#### Action Penalties

The environment penalizes actions that are:
- invalid
- repeated
- explicitly harmful
- strategically lazy, such as doing nothing during a live incident

This matters because in real operations, bad actions are not neutral. They consume time, burn trust, and can deepen instability.

The agent is therefore pushed toward responsible action selection, not brute-force experimentation.

#### Urgency Penalty

The urgency penalty increases over time.

This creates a very important pressure gradient:
- a correct action taken early is more valuable than the same action taken late
- hesitation has a measurable cost
- recovery is judged not just by outcome, but by how long the system suffered before that outcome arrived

This is one of the key reasons the hard task feels realistic. It captures the fact that incident response is always racing against worsening consequences.

#### Progress Rewards

Not every good action fully resolves the system, and the reward design acknowledges that.

The environment grants progress rewards for meaningful improvement such as:
- reducing critical failure severity
- moving the system from worse states toward healthier ones
- improving SLA-related conditions before full completion

This is important because real recovery is rarely instantaneous. Strong agents should be rewarded for moving the system in the right direction even before the final recovery state is reached.

That gives the task a more realistic shape:
- partial recovery matters
- meaningful stabilization matters
- intermediate decisions matter

#### SLA Guardrails

The hard task also includes guardrails and forbidden outcomes.

Some actions may immediately create unacceptable system states or violate the scenario's operational constraints. In those cases, the environment can terminate early with a strong negative outcome.

This models a very real production truth:
- not every technically possible action is operationally acceptable
- some decisions may reduce pressure but still violate business-critical guarantees
- systems are not only protected by engineering goals, but also by service obligations

The reward therefore reflects both technical and organizational reality.

### Why the Three Designs Work Together

Taken together, the reward functions create a progression in operational intelligence:

- `easy` teaches accurate single-step correction
- `medium` teaches diagnostic sequencing and hidden-state reasoning
- `hard` teaches strategic recovery under pressure, cost, and policy constraints

This tiered reward design is powerful because each task adds a new layer of realism without abandoning clarity.

The agent first learns:
- how to avoid obvious mistakes

Then it learns:
- how to reason through incomplete evidence

Finally, it learns:
- how to act when the system is actively deteriorating and every decision carries trade-offs

That progression is what makes the environment more than a benchmark. It makes it a meaningful test of operational reasoning.

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
git clone https://github.com/nithishgouds/OpsSim-AI
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
HF_TOKEN=<your_token_here>
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
```

### 5. Run the Project

```bash
python inference.py
```

### 6. Docker Installation and Setup

Install Docker before building the project container:

- Windows: install Docker Desktop and make sure the Docker engine is running
- Linux: install Docker Engine and verify that the `docker` command works
- WSL: enable Docker Desktop WSL integration if you are running the project from Ubuntu or another WSL distro

Build the Docker image:

```bash
docker build -t opssim-ai .
```

Run the Docker container:

```bash
docker run -p 7860:7860 opssim-ai
```

Optional health check:

```bash
curl http://localhost:7860/health
```

## Example Output

```text
[START] task=hard_scenario_1 env=opssim_ai model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action=restart(database) reward=-0.25 done=false error=null
[STEP] step=2 action=reroute_traffic reward=0.30 done=false error=null
[STEP] step=3 action=scale_backend reward=1.10 done=true error=null
[END] success=true steps=3 rewards=-0.25,0.30,1.10
```
## Baseline Score

```text
[START] task=easy_scenario_1 env=opssim_ai model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action=reboot_cache_cluster reward=0.95 done=true error=null
[END] success=true steps=1 rewards=0.95
[START] task=medium_scenario_1 env=opssim_ai model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action=analyze_failure_timestamps reward=0.27 done=false error=null
[STEP] step=2 action=isolate_date_parsing_logic reward=0.24 done=false error=null
[STEP] step=3 action=deploy_weekend_date_patch reward=0.91 done=true error=null
[END] success=true steps=3 rewards=0.27,0.24,0.91
[START] task=hard_scenario_1 env=opssim_ai model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action=do_nothing reward=-1.05 done=false error=null
[STEP] step=2 action=restart(checkout_cart) reward=-0.80 done=false error=null
[STEP] step=3 action=restart(payment_gateway) reward=-1.35 done=false error=null
[STEP] step=4 action=shutdown(user_analytics) reward=-0.40 done=false error=null
[STEP] step=5 action=shutdown(recommendation_engine) reward=-0.40 done=false error=null
[STEP] step=6 action=restart(checkout_cart) reward=0.75 done=true error=null
[END] success=true steps=6 rewards=-1.05,-0.80,-1.35,-0.40,-0.40,0.75
Final Score = 0.7744
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
