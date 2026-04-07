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

    def _build_easy_prompt(self, obs) -> str:
        return f"""
DO NOT output anything except valid JSON.

You are an automated DevOps recovery agent. Your objective is to resolve a system alert by executing a single configuration fix.

1. Analyze the panicked user message and the specific error logs.
2. Review the current config state.
3. Identify and execute the single correct configuration action that resolves the root cause.
4. BEWARE of "red herring" actions: The user might blame the wrong system (e.g., UI changes) when the logs indicate a deeper infrastructure failure.

User Message: "{obs.user_message}"
Logs: {obs.logs}

Current Config State:
{json.dumps(obs.config, indent=2)}

Available Actions:
{json.dumps(obs.available_actions)}

Return ONLY JSON:
{{
  "action": "<exact_string_from_available_actions>"
}}
"""

    def _parse_easy_response(self, text, obs):
        try:
            data = json.loads(text)
            action = data.get("action", "")
            
            if action not in obs.available_actions:
                action = "do_nothing"
                
            return action, 1.0, None
        except Exception:
            return "do_nothing", 0.0, None

    def _parse_medium(self, obs):
        prompt = self._build_medium_prompt(obs)
        response = self._call_llm(prompt)
        return self._parse_medium_response(response, obs)

    def _build_medium_prompt(self, obs):
        return f"""
DO NOT output anything except valid JSON.

You are an AI agent diagnosing a production issue in a dynamic environment.

- **Logs (Snapshot):** The Logs section now only shows the result of your **very last action**. Use the [IMPACT] and [HINT] provided to understand how the system state changed.
- **User Complaints:** These are your primary signal for overall system health. If complaints remain "unchanged" or "bad," your last action likely didn't address the root cause.
- **Decision Making:** Since you only see the most recent result, maintain an internal logic of your progress. Avoid repeating actions that previously yielded "No meaningful change" or "Analysis already complete."

User complaints:
{obs.user_messages}

Last Action Result:
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
        
def grade_easy(num_scenarios=1):
    total_score = 0.0
    
    env = DevOpsEnv()
    parser = LLMParser()
    
    for i in range(num_scenarios):
        obs = env.reset(task = "easy")
        total_reward = 0.0
        done = False
        rewards_list = []
        action_history = [] 
        
        print(f"[START] task=easy_scenario_{i+1} env=ops-sim model={MODEL_NAME}")
        
        for step in range(MAX_STEPS):
            action_str, _, target = parser.parse(obs, action_history)
            
            error_msg = "null"
            if not obs.available_actions or action_str not in obs.available_actions:
                error_msg = f"invalid_action_{action_str}"
                action_str = "do_nothing"

            obs, reward, done, info = env.step(Action(action_type=action_str, target=target))
            
            log_action = f"{action_str}({target})" if target else action_str
            action_history.append(log_action)
            
            total_reward += reward
            rewards_list.append(f"{reward:.2f}")

            print(f"[STEP] step={step+1} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
            
            if done:
                break

        success = "true" if (done and total_reward > 0) else "false"
        rewards_str = ",".join(rewards_list)

        print(f"[END] success={success} steps={len(rewards_list)} rewards={rewards_str}")

        scenario_score = max(0.0, min(1.0, (total_reward - -1.6) / (1.0 - -1.6)))
        total_score += scenario_score

    return total_score / num_scenarios

def grade_medium(num_scenarios = 1):
    total_score = 0.0
    
    env = DevOpsEnv()
    parser = LLMParser()
    
    for i in range(num_scenarios):
        obs = env.reset(task = "medium")
        total_reward = 0.0
        done = False
        rewards_list = []
        # Add action_history to match the new parse method signature
        action_history = [] 
        
        # [START] line at episode begin [cite: 28]
        print(f"[START] task=medium_scenario_{i+1} env=ops-sim model={MODEL_NAME}")
        
        for step in range(MAX_STEPS):
            # Pass action_history here to fix the error
            action_str, _, target = parser.parse(obs, action_history)
            
            error_msg = "null"
            if not obs.available_actions or action_str not in obs.available_actions:
                error_msg = f"invalid_action_{action_str}"
                action_str = "do_nothing"

            obs, reward, done, info = env.step(Action(action_type=action_str, target=target))
            
            # Update history so the LLM can see its progress in the next step
            log_action = f"{action_str}({target})" if target else action_str
            action_history.append(log_action)
            
            total_reward += reward
            rewards_list.append(f"{reward:.2f}")

            # [STEP] line immediately after env.step() [cite: 29, 31]
            print(f"[STEP] step={step+1} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
            
            if done:
                break

        success = "true" if (done and total_reward > 0) else "false" [cite: 31]
        rewards_str = ",".join(rewards_list)

        # [END] line after episode completion [cite: 30]
        print(f"[END] success={success} steps={len(rewards_list)} rewards={rewards_str}")

        scenario_score = max(0.0, min(1.0, (total_reward - -3.0) / (2.0 - -3.0)))
        total_score += scenario_score

    return total_score / num_scenarios

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
    env = DevOpsEnv( seed=42)
    parser = LLMParser()
    obs = env.reset(task = "hard")
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
    print("Easy Final Score =", grade_easy())
    print("Medium Final Score =", grade_medium())
    print("Hard Final Score =", grade_hard())

if __name__ == "__main__":
    main()