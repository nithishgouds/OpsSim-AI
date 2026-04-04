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

    def parse(self, observation, action_history):
        key = self._build_cache_key(observation)
        if key in self.cache:
            return self.cache[key]

        if observation.task_type == "easy":
            result = self._parse_easy(observation)
        elif observation.task_type == "medium":
            result = self._parse_medium(observation)
        else:
            result = self._parse_hard(observation, action_history)

        if result[0] in observation.available_actions and result[0] != "do_nothing" and result[0]:
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

    def _parse_hard(self, obs, action_history):
        prompt = self._build_hard_prompt(obs, action_history)
        response = self._call_llm(prompt)
        return self._parse_hard_response(response, obs)

    def _build_hard_prompt(self, obs, action_history):
        history_str = "\n".join([f"{i+1}. {act}" for i, act in enumerate(action_history)]) if action_history else "None"
        
        return f"""
DO NOT output anything except valid JSON.

You are managing a catastrophic system failure. You must follow the SLA Playbook rules exactly.

IMPORTANT RULES:
- Do NOT repeat actions that resulted in negative reward.
- Avoid "do_nothing" unless absolutely necessary.
- If an action caused no improvement, try a different strategy.
- Always aim to reduce system failure quickly.
- Choose ONLY from Available Actions.
- Do NOT hallucinate new actions.

Goal:
- Minimize total penalty
- Follow SLA rules strictly
- Stabilize system as fast as possible

Playbook:
{obs.playbook_text}

System State:
{json.dumps(obs.system_state, indent=2)}

Logs (Previous Steps Details):
{obs.logs}

Previous Actions Taken:
{history_str}

Available Actions:
{json.dumps(obs.available_actions)}

Step Count: {obs.step_count}

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
            action = data.get("action", "")
            target = data.get("target", None)
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            
            if not action:
                 return sorted(obs.available_actions)[0], 0.0, None
            
            return action, confidence, target
        except Exception:
            return sorted(obs.available_actions)[0], 0.0, None

    def _call_llm(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=150
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def _build_cache_key(self, obs):
        if obs.task_type == "hard":
            return json.dumps({
                "state": obs.system_state,
                "step": obs.step_count,
                "actions": obs.available_actions
            }, sort_keys=True)
        else:
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
    
    print("[START] medium")
    while not done:
        action_str, _, _ = parser.parse(obs, [])
        obs, reward, done, _ = env.step(Action(action_type=action_str))
        total_reward += reward
        print(f"[STEP] {obs.step_count} | {action_str} | {reward:+.2f}")

    score = max(0, min(1, (total_reward - -3.0) / (2.0 - -3.0)))
    print(f"[END] reward={total_reward:.2f} score={score:.2f}")
    return score

def _calculate_dynamic_min_reward(env: DevOpsEnv, max_steps: int) -> float:
    worst_bleed_per_step = 0.0
    for rule in env.state_data.get("bleed_rules", []):
        penalty = rule.get("penalty", 0.0)
        if penalty < 0:
            worst_bleed_per_step += penalty
            
    worst_invalid_action_penalty = -0.2
    worst_do_nothing_penalty = -0.3
    worst_urgency_penalty = -0.05
    
    worst_action_penalty_per_step = worst_invalid_action_penalty + worst_do_nothing_penalty
    
    n = max_steps
    worst_repeat_penalty_total = -0.15 * (n * (n - 1) / 2)
    worst_repeat_penalty_per_step = worst_repeat_penalty_total / max_steps
    
    worst_case_per_step = (
        worst_bleed_per_step 
        + worst_action_penalty_per_step 
        + worst_urgency_penalty 
        + worst_repeat_penalty_per_step
    )
    
    sla_violation_penalty = env.state_data.get("sla_violation_penalty", -1.0)
    
    min_reward = (max_steps * worst_case_per_step) + sla_violation_penalty
    return min_reward

def grade_hard():
    env = DevOpsEnv(task_type="hard", seed=42)
    parser = LLMParser()
    obs = env.reset()
    total_reward = 0.0
    done = False
    
    min_reward = _calculate_dynamic_min_reward(env, MAX_STEPS)
    max_reward = 1.0
    
    action_history = []
    
    print("[START] hard")
    for step in range(MAX_STEPS):
        action_str, confidence, target = parser.parse(obs, action_history)
        action = Action(action_type=action_str, target=target)
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        log_action = f"{action_str}({target})" if target else action_str
        action_history.append(log_action)
        
        print(f"[STEP] {obs.step_count} | {log_action} | {reward:+.2f}")
        
        if done:
            break

    final_score = (total_reward - min_reward) / (max_reward - min_reward)
    final_score = max(0.0, min(1.0, final_score))
    
    print(f"[END] reward={total_reward:.2f} score={final_score:.2f}")
    return final_score

def main() -> None:
    grade_hard()

if __name__ == "__main__":
    main()