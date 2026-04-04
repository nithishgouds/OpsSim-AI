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
MAX_STEPS = 8

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

You are diagnosing a production issue.

- Logs contain a running history of actions and their effects.
- Also listen to user messages, they highlight potential issues.
- Use them carefully to understand what worked and what did not.
- Avoid repeating actions that show "No meaningful change".

General approach:
- Start by understanding patterns in failures
- Then investigate deeper causes
- Once confident, take corrective action

User complaints:
{obs.user_messages}

Logs:
{obs.logs}

Available actions:
{obs.available_actions}

Return ONLY JSON:
{{
  "action": "<one action>",
  "confidence": 0.9
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
    env = DevOpsEnv(task_type="medium")
    parser = LLMParser()
    obs = env.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        action_str, _, _ = parser.parse(obs)
        obs, reward, done, _ = env.step(Action(action_type=action_str))
        total_reward += reward

    score = max(0, min(1, (total_reward - -3.0) / (2.0 - -3.0)))
    return score

def grade_hard():
    env = DevOpsEnv(task_type="hard", seed=42)
    parser = LLMParser()
    obs = env.reset()
    total_reward = 0.0
    done = False
    
    for step in range(MAX_STEPS):
        action_str, confidence, target = parser.parse(obs)
        
        if action_str not in obs.available_actions:
            action_str = "do_nothing"
            target = None

        action = Action(action_type=action_str, target=target)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    min_reward = -5.0
    max_reward = 1.0
    final_score = (total_reward - min_reward) / (max_reward - min_reward)
    final_score = max(0.0, min(1.0, final_score))
    return final_score

def main() -> None:
    print("Final Score =", grade_hard())

if __name__ == "__main__":
    main()