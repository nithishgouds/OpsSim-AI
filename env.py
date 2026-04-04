import json
import os
import random
import re
from models import Observation, Action

class DevOpsEnv:
    def __init__(self, seed=42, max_steps=8, task_type="hard"):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.task_type = task_type
        self.scenario_index = 0
        self.state_data = {}
        self.observation = None

    def reset(self) -> Observation:
        self.step_count = 0

        if self.task_type == "easy":
            return self._reset_easy()
        elif self.task_type == "medium":
            return self._reset_medium()
        else:
            return self._reset_hard()

    def _reset_easy(self):
        return None

    def _reset_medium(self):
        self.step_count = 0

        with open(os.path.join("tasks", "medium.json"), "r") as f:
            dataset = json.load(f)["medium_tasks_dataset"]

        scenario = dataset[self.scenario_index % len(dataset)]
        self.scenario_index += 1

        self.state_data = {
            "state": scenario["initial_state"].copy(),
            "logs": scenario["observation"]["logs"],
            "user_messages": list(scenario["observation"]["user_messages"]),
            "system_metrics": scenario["observation"]["system_metrics"],
            "action_space": scenario["action_space"],
            "log_templates": scenario["log_templates"],
            "rewards": scenario["dynamic_rewards"],
            "penalties": scenario["dynamic_penalties"],
            "history": []
        }

        return Observation(
            task_type="medium",
            available_actions=self.state_data["action_space"],
            user_messages=self.state_data["user_messages"],
            system_metrics=self.state_data["system_metrics"],
            logs=self.state_data["logs"],
            step_count=self.step_count
        )

    def _reset_hard(self):
        with open(os.path.join("tasks", "hard.json"), "r") as f:
            dataset = json.load(f)["hard_tasks_dataset"]

        scenario = dataset[self.scenario_index % len(dataset)]
        self.scenario_index += 1

        self.state_data = {
            "scenario_id": scenario.get("scenario_id", ""),
            "state": json.loads(json.dumps(scenario.get("initial_state", {}))),
            "penalties": scenario.get("penalties", {}).copy(),
            "optimal_solution_path": scenario.get("optimal_solution_path", []),
            "transition_rules": scenario.get("transition_rules", {}),
            "bleed_rules": scenario.get("bleed_rules", []),
            "sla_rules": scenario.get("sla_rules", {"required": [], "forbidden": []})
        }

        available_actions = list(self.state_data["optimal_solution_path"])
        available_actions.extend(list(self.state_data["penalties"].keys()))
        available_actions.append("do_nothing")
        
        available_actions = sorted(list(set(available_actions)))

        self.observation = Observation(
            task_type="hard",
            available_actions=available_actions,
            system_state=self.state_data["state"],
            playbook_text=scenario.get("playbook_text", ""),
            logs=scenario.get("description", ""),
            step_count=self.step_count
        )
        return self.observation

    def step(self, action: Action):
        self.step_count += 1

        if self.task_type == "easy":
            obs, reward, done, info = self._step_easy(action)
        elif self.task_type == "medium":
            obs, reward, done, info = self._step_medium(action)
        else:
            obs, reward, done, info = self._step_hard(action)

        if obs is not None:
            obs.step_count = self.step_count
            self.observation = obs

        return obs, reward, done, info

    def _step_easy(self, action):
        return None, 0.0, False, {}

    def _step_medium(self, action: Action):
        reward = 0
        done = False

        state = self.state_data["state"]
        logs = self.state_data["logs"]
        history = self.state_data["history"]
        templates = self.state_data["log_templates"]
        rewards_cfg = self.state_data["rewards"]
        penalties_cfg = self.state_data["penalties"]

        action_str = action.action_type
        history.append(action_str)

        prev_state = state.copy()
        prev_health = state["service_health"]

        if not state["pattern_identified"]:
            allowed = ["analyze_failure_pattern"]
        elif not state["root_cause_identified"]:
            allowed = ["identify_edge_case_condition", "inspect_transaction_logs"]
        elif state["bug_active"]:
            allowed = ["apply_transaction_fix"]
        else:
            allowed = []

        allowed += ["restart_payment_service", "do_nothing"]

        if action_str not in allowed:
            reward -= 0.2
            action_str = "do_nothing"

        repeat_count = history.count(action_str)
        if repeat_count > 1:
            reward -= 0.15 * (repeat_count - 1)

        new_log = templates.get(action_str, "")

        if action_str == "analyze_failure_pattern":
            if not state["pattern_identified"]:
                state["pattern_identified"] = True
                state["service_health"] += 0.1
                reward += rewards_cfg["pattern_identified"]
            else:
                reward -= 0.2

        elif action_str == "identify_edge_case_condition":
            if state["pattern_identified"] and not state["root_cause_identified"]:
                state["root_cause_identified"] = True
                state["service_health"] += 0.15
                reward += rewards_cfg["root_cause_identified"]
            else:
                reward -= 0.2

        elif action_str == "apply_transaction_fix":
            if state["root_cause_identified"] and state["bug_active"]:
                state["bug_active"] = False
                state["service_health"] += 0.3
                reward += rewards_cfg["bug_fixed"]
                done = True
            else:
                reward -= 0.3

        elif action_str == "inspect_transaction_logs":
            if not state["root_cause_identified"]:
                state["service_health"] += 0.05
                reward -= 0.02
            else:
                reward -= 0.1

        elif action_str == "restart_payment_service":
            state["service_health"] -= 0.2
            reward += penalties_cfg["restart_without_diagnosis"]

        elif action_str == "do_nothing":
            state["service_health"] -= 0.1
            reward -= 0.1

        state["service_health"] = max(0, min(1, state["service_health"]))

        if state == prev_state:
            reward -= 0.2

        reward -= 0.03 * self.step_count

        if state["root_cause_identified"] and action_str != "apply_transaction_fix":
            reward -= 0.1

        impact = []

        for k in state:
            if k != "service_health" and prev_state[k] != state[k]:
                impact.append(f"{k} → {state[k]}")

        delta = state["service_health"] - prev_health
        if abs(delta) > 1e-6:
            sign = "+" if delta > 0 else ""
            impact.append(f"health {prev_health:.2f}→{state['service_health']:.2f} ({sign}{delta:.2f})")

        if not impact:
            impact.append("No meaningful change")

        if not state["pattern_identified"]:
            hint = "Look for patterns in failures"
        elif not state["root_cause_identified"]:
            hint = "Investigate logs deeper"
        elif state["bug_active"]:
            hint = "A fix is now possible"
        else:
            hint = "System stable"

        self.state_data["logs"] += (
            f"\n--- Step {self.step_count} ---\n"
            f"[ACTION] {action_str}\n"
            f"[IMPACT] {'; '.join(impact)}\n"
            f"[HINT] {hint}\n"
        )

        if done:
            self.state_data["user_messages"] = ["Everything is working perfectly now"]

        else:
            if state["service_health"] > prev_health:
                self.state_data["user_messages"] = [
                    "Seems slightly better",
                    "Still not fully fixed though"
                ]
            elif state["service_health"] < prev_health:
                self.state_data["user_messages"] = [
                    "This is getting worse",
                    "More failures now"
                ]
            else:
                self.state_data["user_messages"] = [
                    "Still inconsistent",
                    "No real improvement"
                ]

        if self.step_count >= self.max_steps:
            done = True

        return Observation(
            task_type="medium",
            available_actions=allowed,
            user_messages=self.state_data["user_messages"],
            system_metrics=self.state_data["system_metrics"],
            logs=self.state_data["logs"],
            step_count=self.step_count
        ), reward, done, {}

    def _step_hard(self, action: Action):
        done = False
        action_str = action.action_type
        if getattr(action, 'target', None):
            action_str = f"{action.action_type}({action.target})"

        state = self.state_data["state"]

        action_penalty = 0.0

        if action_str not in self.observation.available_actions and action_str != "do_nothing":
            action_penalty -= 0.2

        if action_str in self.state_data["penalties"]:
            penalty_val = float(self.state_data["penalties"][action_str])
            action_penalty += penalty_val
            
            if penalty_val <= -0.8:
                self.observation.system_state = state
                self.observation.step_count = self.step_count
                return self.observation, -1.0, True, {"reason": "guardrail_violation"}

        # Store previous state for progress detection
        prev_state = json.loads(json.dumps(state))

        # Fully Data-Driven Transitions
        self._apply_state_transition(state, action_str)

        # Detect positive progress
        progress_reward = 0.0
        if self._detect_positive_progress(prev_state, state):
            progress_reward = 0.1

        # Calculate standard penalties
        bleed_loss = self._calculate_dynamic_bleed(state)
        urgency_penalty = -0.01 * self.step_count
        success_reward = 0.0
        
        # Explicit SLA validation replaces old tradeoff mechanics
        sla_status = self._check_sla_compliance(state)

        if sla_status == "FAIL":
            self.observation.system_state = state
            self.observation.step_count = self.step_count
            return self.observation, -1.0, True, {"reason": "sla_violation"}

        if sla_status == "PASS":
            success_reward = 1.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        step_reward = bleed_loss + action_penalty + urgency_penalty + progress_reward + success_reward

        self.observation.system_state = state
        return self.observation, step_reward, done, {}

    # ==========================================
    # DATA-DRIVEN HARD TASK HELPER FUNCTIONS
    # ==========================================

    def _detect_positive_progress(self, prev_state, new_state) -> bool:
        """Detects if the new state is meaningfully better than the previous state."""
        def extract_values(d, prefix=""):
            vals = {}
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    vals.update(extract_values(v, full_key))
                else:
                    vals[full_key] = v
            return vals

        prev_vals = extract_values(prev_state)
        new_vals = extract_values(new_state)

        # Heuristics to define what constitutes a "bad" vs "good" status string
        negative_terms = ["failing", "degraded", "overloaded", "maxed", "offline", "dropped", "severed"]
        positive_terms = ["online", "healthy", "stable", "complete", "normal", "routed", "restored", "scrubbed_and_stable"]

        for key, new_val in new_vals.items():
            prev_val = prev_vals.get(key)
            if prev_val == new_val or prev_val is None:
                continue

            # 1. Numeric improvement (lower is better for loads, connections, errors, costs)
            def parse_num(v):
                if isinstance(v, (int, float)): return float(v)
                if isinstance(v, str):
                    try:
                        return float(v.replace('%', '').replace('$', '').replace(',', '').strip())
                    except ValueError:
                        return None
                return None

            prev_num = parse_num(prev_val)
            new_num = parse_num(new_val)

            if prev_num is not None and new_num is not None:
                if new_num < prev_num:
                    return True

            # 2. String status improvement
            if isinstance(prev_val, str) and isinstance(new_val, str):
                prev_str = prev_val.lower()
                new_str = new_val.lower()

                was_bad = any(t in prev_str for t in negative_terms)
                is_bad_now = any(t in new_str for t in negative_terms)
                is_good_now = any(t in new_str for t in positive_terms)

                # Check if it moved from a explicitly bad state, or strictly into an explicitly good state
                if (was_bad and not is_bad_now) or (not was_bad and is_good_now):
                    return True

        return False

    def evaluate_condition(self, state, condition_string):
        """Dynamically evaluates conditions like 'services.checkout.status == online'"""
        if not condition_string:
            return True
            
        if " OR " in condition_string:
            return any(self.evaluate_condition(state, c.strip()) for c in condition_string.split(" OR "))
        if " AND " in condition_string:
            return all(self.evaluate_condition(state, c.strip()) for c in condition_string.split(" AND "))

        match = re.match(r"([\w\.]+)\s*(==|!=|<=|>=|<|>|IN)\s*(.+)", condition_string)
        if not match:
            return True
            
        key, op, val_str = match.groups()
        
        parts = key.split('.')
        curr = state
        for p in parts:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
            else:
                curr = None
                break
                
        def parse_numeric(v):
            if isinstance(v, str):
                v = v.replace('$', '').replace(',', '').strip("'").strip('"')
                try:
                    return float(v)
                except ValueError:
                    if v.lower() == 'true': return True
                    if v.lower() == 'false': return False
                    return v
            return v

        curr_val = parse_numeric(curr)
        
        if curr is None:
            return False
            
        if op == "IN":
            target_list = [parse_numeric(v.strip().strip("[]'\"")) for v in val_str.split(",")]
            return curr_val in target_list

        target_val = parse_numeric(val_str)

        if op == "==": return str(curr_val).lower() == str(target_val).lower()
        if op == "!=": return str(curr_val).lower() != str(target_val).lower()
        
        try:
            curr_f = float(curr_val)
            target_f = float(target_val)
            if op == "<=": return curr_f <= target_f
            if op == ">=": return curr_f >= target_f
            if op == "<": return curr_f < target_f
            if op == ">": return curr_f > target_f
        except:
            pass
            
        return False

    def apply_effects(self, state, effects_dict):
        """Dynamically modifies deeply nested state paths via rules dictionary"""
        if not effects_dict:
            return
            
        for key, effect in effects_dict.items():
            parts = key.split('.')
            curr = state
            for p in parts[:-1]:
                if p not in curr:
                    curr[p] = {}
                curr = curr[p]
            target_key = parts[-1]
            
            # Subtraction/Addition handler
            if isinstance(effect, str) and effect.startswith("-") and effect[1:].replace('.', '').isdigit():
                current_val = float(str(curr.get(target_key, 0)).replace('%',''))
                curr[target_key] = max(0.0, current_val - float(effect[1:]))
            elif isinstance(effect, str) and effect.startswith("+") and effect[1:].replace('.', '').isdigit():
                current_val = float(str(curr.get(target_key, 0)).replace('%',''))
                curr[target_key] = current_val + float(effect[1:])
            else:
                curr[target_key] = effect

    def _apply_state_transition(self, state, action_str):
        rules = self.state_data.get("transition_rules", {})
        if action_str in rules:
            rule = rules[action_str]
            condition = rule.get("condition", "")
            if self.evaluate_condition(state, condition):
                self.apply_effects(state, rule.get("effects", {}))
            else:
                if "else_effects" in rule:
                    self.apply_effects(state, rule["else_effects"])

    def _calculate_dynamic_bleed(self, state) -> float:
        bleed = 0.0
        for rule in self.state_data.get("bleed_rules", []):
            if self.evaluate_condition(state, rule.get("condition", "")):
                bleed += float(rule.get("penalty", 0.0))
        return bleed

    def _check_sla_compliance(self, state) -> str:
        """Evaluates system state against required business logic/guardrails"""
        sla_rules = self.state_data.get("sla_rules", {})
        forbidden_rules = sla_rules.get("forbidden", [])
        required_rules = sla_rules.get("required", [])

        # 1. If ANY forbidden condition is TRUE -> FAIL immediately
        for cond in forbidden_rules:
            if self.evaluate_condition(state, cond):
                return "FAIL"

        # 2. If ALL required conditions are TRUE -> PASS
        if required_rules:
            all_passed = all(self.evaluate_condition(state, cond) for cond in required_rules)
            if all_passed:
                return "PASS"
        
        # 3. Else -> INCOMPLETE
        return "INCOMPLETE"

    def state(self):
        return {
            "task_type": self.task_type,
            "state": self.state_data,
            "step_count": self.step_count
        }