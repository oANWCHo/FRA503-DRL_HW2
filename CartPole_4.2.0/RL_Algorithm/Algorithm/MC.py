from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType
import torch
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
        
    def update(
        self,
        
    ):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        """
        return_sum = 0 

        
        # Update First Visit Monte Carlo
        for t in reversed(range(len(self.obs_hist))):

            state_tensor = self.obs_hist[t].get('policy')  # แก้ตรงนี้ให้เป็นคีย์ที่ถูกต้อง

            # แปลง Tensor → Numpy → Tuple เพื่อให้เป็น hashable key
            state = tuple(state_tensor.flatten().tolist())  
            action = self.action_hist[t]
            reward = self.reward_hist[t]
            return_sum = self.discount_factor * return_sum + reward  # Compute return
            
            # print([state,action,reward])
             # ตรวจสอบว่าค่าของ state มีอยู่ใน Q-table หรือไม่
            if state not in self.q_values:
                self.q_values[state] = torch.zeros(self.num_of_action)
                self.n_values[state] = torch.zeros(self.num_of_action)

            # ตรวจสอบเฉพาะค่าที่เป็น `Tensor` ก่อนใช้ `torch.equal()`
            def tensor_equal(tensor1, tensor2):
                
                if isinstance(tensor2, dict) and 'policy' in tensor2:
                    tensor2 = tensor2.get('policy')  # ดึงค่าถ้าเป็น dict

                # print([tensor1,tensor2])
                return isinstance(tensor2, torch.Tensor) and torch.equal(tensor1, tensor2)
        
            if not any(tensor_equal(state_tensor, s) for s in self.obs_hist[:t]):  # First-visit MC update
                # print("New!")
                self.n_values[state][action] += 1
                self.q_values[state][action] += (return_sum - self.q_values[state][action]) / self.n_values[state][action]
            # else:
            #     # print("no")