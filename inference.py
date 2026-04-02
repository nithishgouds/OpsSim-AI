import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from env import DevOpsEnv
from models import Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

class LLMParser:
    def __init__(self):
        self.client = client
        self.model = MODEL_NAME
        self.cache = {}

    def parse(self, observation):
        key = self._build_cache_key(observation)
        if key in self.cache:
            return self.cache[key]

        if observation.task_type == "easy":
            result = self._parse_easy(observation)
        elif observation.task_type == "medium":
            result = self._parse_medium(observation)
        else:
            result = self._parse_hard(observation)

        self.cache[key] = result
        return result

    def _parse_easy(self, obs):
        prompt = self._build_easy_prompt(obs)
        response = self._call_llm(prompt)
        return self._parse_easy_response(response, obs)

    def _build_easy_prompt(self, obs):
        return ""

    def _parse_easy_response(self, text, obs):
        return "do_nothing", 0.0, None

    def _parse_medium(self, obs):
        prompt = self._build_medium_prompt(obs)
        # print("="*50,"\nPrompt:\n",prompt)
        response = self._call_llm(prompt)
        return self._parse_medium_response(response, obs)

    def _build_medium_prompt(self, obs):
        return f"""
DO NOT output anything except valid JSON.

You are diagnosing a subtle production issue.

IMPORTANT CONTEXT:
- System metrics appear NORMAL (CPU, error rate etc.)
- The issue is hidden and must be inferred from USER COMPLAINTS and LOG PATTERNS
- The system evolves after each step:
    • Logs will change
    • User complaints will become better or worse
    • System health improves or degrades based on your actions

STRATEGY:
- Do NOT jump to fixes immediately
- First identify patterns from complaints
- Then infer root cause from logs
- Then apply fix
- Incorrect or random actions will worsen the system
- Avoid repeating the same action unless the system state changes.

User complaints:
{obs.user_messages}

Logs:
{obs.logs}

System metrics:
{obs.system_metrics}

Available actions:
{obs.available_actions}

Step count: {obs.step_count}

Goal:
Choose the NEXT BEST ACTION that moves the system closer to resolution.

Return ONLY JSON:
{{
    "action": "<one of available_actions>",
    "confidence": 0.0-1.0
}}
"""
    def _parse_medium_response(self, text, obs):
        try:
            data = json.loads(text)
            action = data.get("action", "do_nothing")
            confidence = float(data.get("confidence", 0.3))
            if action not in obs.available_actions:
                action = "do_nothing"
            confidence = max(0.0, min(1.0, confidence))
            return action, confidence, None
        except Exception:
            return "do_nothing", 0.3, None

    def _parse_hard(self, obs):
        prompt = self._build_hard_prompt(obs)
        response = self._call_llm(prompt)
        return self._parse_hard_response(response, obs)

    def _build_hard_prompt(self, obs):
        return f"""
DO NOT output anything except valid JSON.

You are managing a catastrophic system failure. You must follow the SLA Playbook rules exactly.

Playbook:
{obs.playbook_text}

System State:
{json.dumps(obs.system_state, indent=2)}

Logs:
{obs.logs}

Available Actions:
{json.dumps(obs.available_actions)}

Step Count: {obs.step_count}

Goal: Choose the exact string from Available Actions to execute next. 
Do not hallucinate actions.

Return ONLY JSON:
{{
    "action": "<exact_string_from_available_actions>",
    "target": null,
    "confidence": 0.0-1.0
}}
"""

    def _parse_hard_response(self, text, obs):
        try:
            data = json.loads(text)
            action = data.get("action", "do_nothing")
            target = data.get("target", None)
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            return action, confidence, target
        except Exception:
            return "do_nothing", 0.0, None

    def _call_llm(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def _build_cache_key(self, obs):
        return json.dumps({
            "task": obs.task_type,
            "messages": getattr(obs, "user_messages", None),
            "logs": getattr(obs, "logs", ""),
            "metrics": getattr(obs, "system_metrics", None),
            "state": getattr(obs, "system_state", None)
        }, sort_keys=True)

def grade_medium():
    """
    Runs one medium task episode and returns final normalized score.
    Uses LLM to choose actions step-by-step.
    """

    env = DevOpsEnv(task_type="medium")
    parser = LLMParser()

    obs = env.reset()
    total_reward = 0.0
    done = False

    print("[START] task=medium env=devops")

    while not done:
        # -------------------------------------
        # LLM decides next action
        # -------------------------------------
        action_str, confidence, _ = parser.parse(obs)

        action = Action(action_type=action_str)

        # -------------------------------------
        # Step environment
        # -------------------------------------
        obs, reward, done, info = env.step(action)

        total_reward += reward

        # -------------------------------------
        # Logging (hackathon format style)
        # -------------------------------------
        print(
            f"[STEP] step={obs.step_count} action={action_str} "
            f"reward={reward:+.2f} done={str(done).lower()} error=null"
        )

        if done:
            break

    # -------------------------------------
    # Normalize score (same logic as hard)
    # -------------------------------------
    min_reward = -3.0
    max_reward = 2.0

    final_score = (total_reward - min_reward) / (max_reward - min_reward)
    final_score = max(0.0, min(1.0, final_score))

    print(
        f"[END] success={str(done).lower()} "
        f"steps={obs.step_count} rewards={total_reward:.2f}"
    )

    return final_score


def grade_hard():
    env = DevOpsEnv(task_type="hard", seed=None)
    parser = LLMParser()
    
    obs = env.reset()
    total_reward = 0.0
    done = False
    
    print("Starting Inference on HARD Task...")
    
    while not done:
        action_str, confidence, target = parser.parse(obs)
        action = Action(action_type=action_str, target=target)
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {obs.step_count}: Action: {action_str} | Reward: {reward:+.2f}")
        
        if done:
            print("Episode complete.")
            if "reason" in info:
                print(f"Termination Reason: {info['reason']}")
            break

    min_reward = -5.0
    max_reward = 1.0
    final_score = (total_reward - min_reward) / (max_reward - min_reward)
    final_score = max(0.0, min(1.0, final_score))

    print(f"Total Reward: {total_reward:+.2f}")
    print(f"Final Score: {final_score:.4f}")

def main() -> None:
    print("Final Score = ",grade_medium())

if __name__ == "__main__":
    main()