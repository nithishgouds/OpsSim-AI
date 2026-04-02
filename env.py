import json
import os
from models import Observation, Action

class DevOpsEnv:
    def __init__(self, seed=42, max_steps=8, task_type="hard"):
        self.max_steps = max_steps
        self.task_type = task_type
        self.scenario_index = 0  # Deterministic sequential indexing
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
        self.state_data["action_history"] = []

        with open(os.path.join("tasks", "medium.json"), "r") as f:
            dataset = json.load(f)["medium_tasks_dataset"]

        # Deterministic selection 
        scenario = dataset[self.scenario_index % len(dataset)]
        self.scenario_index += 1

        self.state_data = {
            "scenario_id": scenario["scenario_id"],

            # Core system state
            "state": {
                "bug_active": scenario["initial_state"]["bug_active"],
                "service_health": scenario["initial_state"]["service_health"],
                "pattern_identified": scenario["initial_state"]["pattern_identified"],
                "root_cause_identified": scenario["initial_state"]["root_cause_identified"],
                "fix_applied": scenario["initial_state"]["fix_applied"]
            },

            # Static configs
            "action_space": scenario["action_space"],
            "dynamic_rewards": scenario["dynamic_rewards"],
            "dynamic_penalties": scenario["dynamic_penalties"],
            "state_transitions": scenario["state_transitions"],
            "log_templates": scenario["log_templates"],
            "user_message_updates": scenario["user_message_updates"],
            "optimal_solution_path": scenario["optimal_solution_path"],

            # Dynamic evolving fields
            "logs": scenario["observation"]["logs"],
            "user_messages": list(scenario["observation"]["user_messages"]),
            "system_metrics": scenario["observation"]["system_metrics"]
        }

        self.observation = Observation(
            task_type="medium",
            available_actions=self.state_data["action_space"],
            user_messages=self.state_data["user_messages"],
            system_metrics=self.state_data["system_metrics"],
            logs=self.state_data["logs"],
            step_count=self.step_count
        )

        return self.observation

    def _reset_hard(self):
        with open(os.path.join("tasks", "hard.json"), "r") as f:
            dataset = json.load(f)["hard_tasks_dataset"]

        # 5. DETERMINISM: Iterate sequentially instead of randomly
        scenario = dataset[self.scenario_index % len(dataset)]
        self.scenario_index += 1

        self.state_data = {
            "scenario_id": scenario.get("scenario_id", ""),
            "state": json.loads(json.dumps(scenario.get("initial_state", {}))),
            "penalties": scenario.get("penalties", {}).copy(),
            "optimal_solution_path": scenario.get("optimal_solution_path", []),
        }

        # Dynamically populate available actions
        available_actions = list(self.state_data["optimal_solution_path"])
        available_actions.extend(list(self.state_data["penalties"].keys()))
        available_actions.append("do_nothing")
        
        # Deduplicate and sort for deterministic output
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

        done = False
        reward = -0.05  # base step cost

        action_str = action.action_type

        state = self.state_data["state"]
        rewards_cfg = self.state_data["dynamic_rewards"]
        penalties_cfg = self.state_data["dynamic_penalties"]
        log_templates = self.state_data["log_templates"]
        user_updates = self.state_data["user_message_updates"]

        history = self.state_data.setdefault("action_history", [])
        history.append(action_str)

        prev_health = state["service_health"]
        prev_state_snapshot = state.copy()

        new_log = ""

        # -----------------------------------------
        # VALIDATE ACTION
        # -----------------------------------------
        if action_str not in self.state_data["action_space"]:
            reward += penalties_cfg.get("irrelevant_action", -0.1)
            action_str = "do_nothing"

        # -----------------------------------------
        # MILD REPEAT PENALTY
        # -----------------------------------------
        repeat_count = history.count(action_str)
        repeat_penalty = 0.0

        if repeat_count > 1:
            repeat_penalty = -0.1 * (repeat_count - 1)
            reward += repeat_penalty

        # -----------------------------------------
        # GATED TRANSITIONS
        # -----------------------------------------

        # STEP 1
        if action_str == "analyze_failure_pattern":
            if not state["pattern_identified"]:
                state["pattern_identified"] = True
                state["service_health"] += 0.1
                reward += rewards_cfg.get("pattern_identified", 0.25)
                new_log = log_templates.get(action_str, "")
            else:
                reward -= 0.05

        # STEP 2
        elif action_str == "identify_edge_case_condition":
            if state["pattern_identified"]:
                if not state["root_cause_identified"]:
                    state["root_cause_identified"] = True
                    state["service_health"] += 0.15
                    reward += rewards_cfg.get("root_cause_identified", 0.4)
                    new_log = log_templates.get(action_str, "")
                else:
                    reward -= 0.05
            else:
                reward += penalties_cfg.get("irrelevant_action", -0.1)

        # STEP 3 (FIX)
        elif action_str == "apply_transaction_fix":
            if state["root_cause_identified"]:
                if state["bug_active"]:
                    state["bug_active"] = False
                    state["fix_applied"] = True
                    state["service_health"] += 0.3

                    reward += rewards_cfg.get("bug_fixed", 1.0)

                    # Early completion bonus
                    reward += max(0, 0.3 - 0.05 * self.step_count)

                    done = True
                    new_log = log_templates.get(action_str, "")
                else:
                    reward -= 0.05
            else:
                reward += penalties_cfg.get("random_fix_attempt", -0.2)

        # OPTIONAL
        elif action_str == "inspect_transaction_logs":
            if not state["root_cause_identified"]:
                state["service_health"] += 0.05
                reward -= 0.02
            else:
                reward -= 0.1  # discourage after root cause
            new_log = log_templates.get(action_str, "")

        # BAD ACTION
        elif action_str == "restart_payment_service":
            state["service_health"] -= 0.2
            reward += penalties_cfg.get("restart_without_diagnosis", -0.3)
            new_log = log_templates.get(action_str, "")

        elif action_str == "do_nothing":
            state["service_health"] -= 0.1
            reward -= 0.1
            new_log = log_templates.get(action_str, "")

        else:
            reward += penalties_cfg.get("irrelevant_action", -0.1)

        # Clamp
        state["service_health"] = max(0.0, min(1.0, state["service_health"]))

        # -----------------------------------------
        # NO PROGRESS (SMALL PENALTY)
        # -----------------------------------------
        if state == prev_state_snapshot:
            reward -= 0.1

        # -----------------------------------------
        # FORCE COMPLETION (LIGHT)
        # -----------------------------------------
        if state["root_cause_identified"] and action_str != "apply_transaction_fix":
            reward -= 0.1

        # -----------------------------------------
        # LOG UPDATE
        # -----------------------------------------
        self.state_data["logs"] += f"\n[ACTION] {action_str}"
        self.state_data["logs"] += "\n" + new_log

        if state["service_health"] > prev_health:
            self.state_data["logs"] += "\n" + log_templates.get("improvement", "")
        elif state["service_health"] < prev_health:
            self.state_data["logs"] += "\n" + log_templates.get("degradation", "")

        # -----------------------------------------
        # USER SENTIMENT
        # -----------------------------------------
        if done:
            self.state_data["user_messages"] = user_updates.get("fixed", self.state_data["user_messages"])
        else:
            if state["service_health"] > prev_health:
                self.state_data["user_messages"] = user_updates.get("improve", self.state_data["user_messages"])
            elif state["service_health"] < prev_health:
                self.state_data["user_messages"] = user_updates.get("degrade", self.state_data["user_messages"])

        # -----------------------------------------
        # TERMINATION
        # -----------------------------------------
        if not state["bug_active"]:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # -----------------------------------------
        # DEBUG
        # -----------------------------------------
        print("\n========== STEP DEBUG ==========")
        print(f"Step: {self.step_count}")
        print(f"Action: {action_str}")
        print(f"Repeat Count: {repeat_count} (Penalty: {repeat_penalty:+.2f})")
        print(f"Service Health: {prev_health:.2f} → {state['service_health']:.2f}")
        print(f"Pattern: {state['pattern_identified']} | Root Cause: {state['root_cause_identified']}")
        print(f"Bug Active: {state['bug_active']}")
        print(f"Reward: {reward:+.2f}")
        print("================================\n")

        obs = Observation(
            task_type="medium",
            available_actions=self.state_data["action_space"],
            user_messages=self.state_data["user_messages"],
            system_metrics=self.state_data["system_metrics"],
            logs=self.state_data["logs"],
            step_count=self.step_count
        )

        return obs, reward, done, {}

    def _step_hard(self, action: Action):
        done = False
        action_str = action.action_type
        if getattr(action, 'target', None):
            action_str = f"{action.action_type}({action.target})"

        state = self.state_data["state"]
        sid = self.state_data["scenario_id"]

        # ==========================================
        # 6. REWARD STRUCTURE - Part 2: Action Penalty
        # ==========================================
        action_penalty = 0.0

        # Action Validation Penalty
        if action_str not in self.observation.available_actions and action_str != "do_nothing":
            action_penalty -= 0.2

        # Exact Penalty Matching & Guardrail Enforcement
        if action_str in self.state_data["penalties"]:
            penalty_val = float(self.state_data["penalties"][action_str])
            action_penalty += penalty_val
            
            # 2. SLA PRIORITY: Immediate termination on guardrail violation
            if penalty_val <= -0.8:
                self.observation.system_state = state
                self.observation.step_count = self.step_count
                return self.observation, -1.0, True, {"reason": "guardrail_violation"}

        # ==========================================
        # STATE TRANSITIONS (Cause and Effect)
        # ==========================================
        self._apply_state_transition(sid, state, action_str)

        # ==========================================
        # 6. REWARD STRUCTURE - Part 1: Dynamic Bleed
        # 1. DYNAMIC BLEED: Based strictly on current state
        # ==========================================
        bleed_loss = self._calculate_dynamic_bleed(sid, state)

        # ==========================================
        # 3. URGENCY PRESSURE: Time-based penalty
        # ==========================================
        urgency_penalty = -0.01 * self.step_count

        # ==========================================
        # 4. EPISODE TERMINATION & 6. Success Reward
        # ==========================================
        success_reward = 0.0
        
        # Episode ends ONLY when all bleeds are cleared (system stabilized)
        if bleed_loss == 0.0:
            success_reward = 1.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # Final Reward Calculation
        step_reward = bleed_loss + action_penalty + urgency_penalty + success_reward

        self.observation.system_state = state
        return self.observation, step_reward, done, {}

    def _apply_state_transition(self, sid, state, action_str):
        """Applies exact state changes based on the agent's action."""
        
        # E-Commerce Black Friday Scenario
        if sid == "hard_001_ecommerce_black_friday":
            if action_str == "shutdown(user_analytics)":
                if state["services"]["user_analytics"]["status"] != "offline":
                    state["services"]["user_analytics"]["status"] = "offline"
                    state["overall_cpu_load"] = max(0, state["overall_cpu_load"] - 30)
            elif action_str == "shutdown(recommendation_engine)":
                if state["services"]["recommendation_engine"]["status"] != "offline":
                    state["services"]["recommendation_engine"]["status"] = "offline"
                    state["overall_cpu_load"] = max(0, state["overall_cpu_load"] - 60)
            elif action_str == "restart(checkout_cart)":
                if state["overall_cpu_load"] <= 100:
                    state["services"]["checkout_cart"]["status"] = "online"

        # Security Exfiltration Scenario
        elif sid == "hard_002_security_data_exfiltration":
            if action_str == "drop_network_interface(internal_db)":
                state["internal_db_outbound_connections"] = 0
            elif action_str == "notify_security_team(severity=critical)":
                if state.get("internal_db_outbound_connections", 0) == 0:
                    state["public_api_status"] = "scrubbed_and_stable"

        # Cloud Runaway Spend Scenario
        elif sid == "hard_003_cloud_runaway_spend":
            if action_str == "freeze_queue(async_batch_queue)":
                state["async_batch_queue"] = "frozen"
            elif action_str == "terminate_instances(type=gpu, exclude_tag=production_inference)":
                state["gpu_nodes_active"] = 10
                state["hourly_burn_rate"] = "$1,500"
            elif action_str == "scale_up(core_auth_api, nodes=5)":
                state["core_auth_api"] = "healthy"
                state["standard_nodes_active"] = 15

        # Split Brain Database Scenario
        elif sid == "hard_004_split_brain_database":
            if action_str == "check_replication_status()":
                state["replication_lag"] = "verified_broken"
            elif action_str == "set_db_mode(db_us_east, mode=read_only)":
                state["db_us_east_status"] = "read_only"
            elif action_str == "page_network_engineers()":
                if state.get("db_us_east_status") == "read_only":
                    state["network_link_us_eu"] = "restored"
                    state["db_us_east_status"] = "accepting_writes"

        # Triage SLA Playbook (Prompt Custom Scenario Support)
        elif sid == "hard_000_enterprise_sla" or "server_load" in state:
            if action_str == "shutdown_service(recommendation_engine)":
                state["recommendation_engine"] = "Offline"
                state["server_load"] = "110%"
                state["enterprise_api_status"] = "Stable"
            elif action_str == "throttle_traffic(free_tier, limit=50%)":
                state["server_load"] = "80%"
                state["free_api_status"] = "Stable"


    def _calculate_dynamic_bleed(self, sid, state) -> float:
        """Calculates dynamic bleed loss strictly based on current state attributes."""
        bleed = 0.0

        if sid == "hard_001_ecommerce_black_friday":
            if state["services"]["payment_gateway"].get("status") in ["failing", "degraded"]:
                bleed -= 0.20
            if state["services"]["checkout_cart"].get("status") == "failing":
                bleed -= 0.15
            if state.get("overall_cpu_load", 0) > 100:
                bleed -= 0.05

        elif sid == "hard_002_security_data_exfiltration":
            if state.get("public_api_status") == "failing":
                bleed -= 0.05
            if state.get("internal_db_outbound_connections", 0) > 0:
                bleed -= 0.30

        elif sid == "hard_003_cloud_runaway_spend":
            burn_rate_str = str(state.get("hourly_burn_rate", "$0")).replace("$", "").replace(",", "")
            burn_rate = int(burn_rate_str)
            if burn_rate > 2000:
                bleed -= 0.15
            if state.get("core_auth_api") == "throttling":
                bleed -= 0.10

        elif sid == "hard_004_split_brain_database":
            # Active Split Brain is disastrous
            if state.get("db_us_east_status") == "accepting_writes" and state.get("db_eu_west_status") == "accepting_writes":
                if state.get("network_link_us_eu") == "severed":
                    bleed -= 0.25
            # US East offline/read-only has a small continuous business cost until network restores
            if state.get("db_us_east_status") == "read_only" and state.get("network_link_us_eu") == "severed":
                bleed -= 0.05
                
        # Triage SLA Playbook (Prompt Custom Scenario Support)
        elif sid == "hard_000_enterprise_sla" or "server_load" in state:
            if state.get("enterprise_api_status") == "Failing":
                bleed -= 0.10
            if state.get("free_api_status") == "Failing":
                bleed -= 0.01

        return bleed

    def state(self):
        return {
            "task_type": self.task_type,
            "state": self.state_data,
            "step_count": self.step_count
        }