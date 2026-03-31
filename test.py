"""
test.py

Simple evaluation script for DevOpsEnv.

Policy:
- Uses LLM suggestion from environment (implicit via reward shaping)
- Falls back to valid random action if needed

Goal:
- Demonstrate environment interaction
- Not optimized, but meaningful baseline
"""

from env import DevOpsEnv
from models import Action
import random


def choose_action(obs):
    """
    Simple heuristic policy:
    - If EASY → try enable_payment
    - If MEDIUM → follow reasoning chain
    - If HARD → shutdown analytics first
    """

    if obs.task_type == "easy":
        return Action(action_type="enable_payment")

    elif obs.task_type == "medium":
        # simple staged reasoning
        if "analyze_complaints" in obs.available_actions:
            return Action(action_type="analyze_complaints")
        elif "identify_service" in obs.available_actions:
            return Action(action_type="identify_service")
        else:
            return Action(action_type="fix_weekend_bug")

    elif obs.task_type == "hard":
        return Action(action_type="shutdown_service", target="analytics")

    # fallback (should not happen)
    return Action(action_type=random.choice(obs.available_actions))


def run_episode(env):
    obs = env.reset()

    print("\n==============================")
    print(f"TASK TYPE: {obs.task_type.upper()}")
    print("==============================")
    print("Initial Observation:\n", obs)

    total_reward = 0

    for step in range(env.max_steps):
        action = choose_action(obs)

        obs, reward, done, _ = env.step(action)

        total_reward += reward.value

        print(f"\nStep {step + 1}")
        print("Action:", action)
        print("Reward:", reward.value)
        print("Done:", done)

        if done:
            break

    print("\nFinal Score:", round(total_reward, 3))
    print("==============================\n")


if __name__ == "__main__":
    env = DevOpsEnv()

    # run multiple episodes
    for _ in range(3):
        run_episode(env)