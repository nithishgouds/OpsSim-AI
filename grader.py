from env import DevOpsEnv
from models import Action
from llm_parser import LLMParser

def run_hard():
    env = DevOpsEnv(task_type="hard")
    parser = LLMParser()

    obs = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action_str, confidence, target = parser.parse(obs)
        action = Action(action_type=action_str, target=target)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    min_reward = -5.0
    max_reward = 1.0
    score = (total_reward - min_reward) / (max_reward - min_reward)
    score = max(0.0, min(1.0, score))

    return total_reward, score


def run_easy():
    return 0.0, 0.0


def run_medium():
    return 0.0, 0.0


if __name__ == "__main__":
    hard_reward, hard_score = run_hard()
    print("HARD TASK")
    print("Total Reward:", hard_reward)
    print("Final Score:", hard_score)