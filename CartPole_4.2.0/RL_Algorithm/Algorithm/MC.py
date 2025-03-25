from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType
import torch
from collections import defaultdict
class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        



    # def update(self):
    #     """
    #     อัปเดตค่า Q-values โดยใช้ Monte Carlo (First-Visit MC)
    #     """
    #     returns_sum = defaultdict(lambda: torch.zeros(self.num_of_action, dtype=torch.float32))
    #     N = defaultdict(lambda: torch.zeros(self.num_of_action, dtype=torch.float32))

    #     for episode_idx in reversed(range(len(self.obs_hist))):
    #         # print(f"\n[DEBUG] Episode {episode_idx}")

    #         # **1. ดึงค่า state และแปลงจาก dict**
    #         state_tensor = self.obs_hist[episode_idx].get('policy', None)
    #         if state_tensor is None:
    #             # print(f"[ERROR] Episode {episode_idx}: Missing 'policy' in obs_hist!")
    #             continue
    #         states = [tuple(torch.round(state_tensor.flatten(), decimals=2).tolist())]  # **ต้องอยู่ใน list**

    #         # **2. แปลง Action จาก Tensor -> List**
    #         actions = self.action_hist[episode_idx]
    #         if isinstance(actions, torch.Tensor):
    #             actions = actions.tolist()

    #         # **3. แปลง Reward จาก Float -> List**
    #         rewards = self.reward_hist[episode_idx]
    #         if isinstance(rewards, float):
    #             rewards = [rewards]  # **ต้องเป็น list**

    #         # ตรวจสอบว่า States, Actions และ Rewards มีขนาดตรงกัน
    #         if len(states) != len(actions) or len(actions) != len(rewards):
    #             # print(f"[ERROR] Episode {episode_idx}: Mismatched Data Sizes! (States: {len(states)}, Actions: {len(actions)}, Rewards: {len(rewards)})")
    #             continue

    #         # คำนวณ Discount Factor
    #         discounts = torch.tensor([self.discount_factor**i for i in range(len(rewards))], dtype=torch.float32)
    #         visited_states_actions = set()

    #         for t in range(len(states)):
    #             state = states[t]
    #             action = actions[t]
    #             G = sum(torch.tensor(rewards[t:]) * discounts[:len(rewards)-t])  # คำนวณ Return

    #             if (state, action) not in visited_states_actions:
    #                 visited_states_actions.add((state, action))
    #                 N[state][action] += 1
    #                 returns_sum[state][action] += G
    #                 self.q_values[state][action] = returns_sum[state][action] / N[state][action]

    #                 # print(f"[DEBUG] Updated Q[{state}, {action}] = {self.q_values[state][action]}")

    def update(self):
            """
            Update Q-values using Monte Carlo.

            This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
            """
            return_sum = 0 

            obs_hist_list = [tuple(obs.cpu().numpy()) if isinstance(obs, torch.Tensor) else tuple(obs) for obs in self.obs_hist]
            # update First occur
            for t in reversed(range(len(self.obs_hist))):
                state = self.obs_hist[t]
                action = self.action_hist[t]
                reward = self.reward_hist[t]
                if isinstance(state, dict):
                    state = tuple(state.values())  
                if isinstance(state, torch.Tensor):
                    state = tuple(state.cpu().numpy()) 

                return_sum = self.discount_factor * return_sum + reward  # Compute return
                
                if state not in obs_hist_list[:t]:  # First-visit MC update
                    self.n_values[state][action] += 1
                    self.q_values[state][action] = ((self.q_values[state][action] * (self.n_values[state][action])) + return_sum) / (self.n_values[state][action] + 1)
                    # self.q_values[state][action] += (return_sum - self.q_values[state][action]) / self.n_values[state][action]




        # def update(self, state_history, action_history, reward_history):
        #     """
        #     Update Q-values using Monte Carlo.

        #     This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        #     """
        #     G = 0

        #     for t in reversed(range(len(self.reward_hist))):
        #         state = state_history[t]
        #         action = int(action_history[t])
        #         G = reward_history[t] + self.discount_factor * G

        #         # Check if the state-action pair is first visit
        #         if state not in state_history[:t]:
        #             self.q_values[state][action] = ((self.q_values[state][action] * (self.n_values[state][action])) + G) / (self.n_values[state][action] + 1)
        #             self.n_values[state][action] += 1