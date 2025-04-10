from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class Double_Q_Learning(BaseAlgorithm):
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
        Initialize the Double Q-Learning algorithm.

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
            control_type=ControlType.DOUBLE_Q_LEARNING,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(self, state, action, reward, next_state):
        """
        Update Q-values using Double Q-Learning.

        This method applies the Double Q-Learning update rule to improve policy decisions by updating the Q-table.
        """

        if next_state not in self.qa_values:
            self.qa_values[next_state] = np.zeros(self.num_of_action)
        if next_state not in self.qb_values:
            self.qb_values[next_state] = np.zeros(self.num_of_action)

        if np.random.rand() < 0.5:
            # Update Q_A using Q_B
            best_next_action = np.argmax(self.qa_values[next_state])
            q_next = 0 if next_state is None else self.qb_values[next_state][best_next_action]
            self.qa_values[state][action] += self.lr * (reward + self.discount_factor * q_next - self.qa_values[state][action])
        else:
            # Update Q_B using Q_A
            best_next_action = np.argmax(self.qb_values[next_state])
            q_next = 0 if next_state is None else self.qa_values[next_state][best_next_action]
            self.qb_values[state][action] += self.lr * (reward + self.discount_factor * q_next - self.qb_values[state][action])
        
        #อัปเดตค่า Q-Table 
        self.q_values[state] = (self.qa_values[state] + self.qb_values[state]) / 2

        