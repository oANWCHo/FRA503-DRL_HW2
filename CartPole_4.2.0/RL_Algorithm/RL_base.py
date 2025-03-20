import torch
from collections import defaultdict
from enum import Enum
import numpy as np
import os
import json
import torch

class ControlType(Enum):
    MONTE_CARLO = 1
    TEMPORAL_DIFFERENCE = 2
    Q_LEARNING = 3
    DOUBLE_Q_LEARNING = 4


class BaseAlgorithm():
    def __init__(
        self,
        control_type: ControlType,
        num_of_action: int,
        action_range: list,  # [min, max]
        discretize_state_weight: list,  # [bins for pose_cart, pose_pole, vel_cart, vel_pole]
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        self.control_type = control_type
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range

        # Define bins as torch tensors for discretization
        # self.bins = [
        #     # torch.linspace(-3, 3, discretize_state_weight[0]),  # pose_cart bins
        #     # torch.linspace(-10, 10, discretize_state_weight[1]),  # pose_pole bins
        #     # torch.linspace(-2, 2, discretize_state_weight[2]),  # vel_cart bins
        #     # torch.linspace(-2, 2, discretize_state_weight[3])  # vel_pole bins
        #     torch.linspace(-3, 3, discretize_state_weight[0]),  # pose_cart bins
        #     torch.linspace(-20, 20, discretize_state_weight[1]),  # pose_pole bins
        #     torch.linspace(-10, 10, discretize_state_weight[2]),  # vel_cart bins
        #     torch.linspace(-10, 10, discretize_state_weight[3])  # vel_pole bins
        # ]
        
        self.discretize_state_weight =  discretize_state_weight

        self.q_values = defaultdict(lambda: torch.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: torch.zeros(self.num_of_action))
        self.training_error = []

        if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: torch.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: torch.zeros(self.num_of_action))

    # def discretize_state(self, obs: dict):
    #     """
    #     Discretize the observation state using predefined bins.

    #     Args:
    #         obs (dict): Observation dictionary.

    #     Returns:
    #         Tuple[int, int, int, int]: Discretized state.
    #     """
    #     required_keys = ['pose_cart', 'pose_pole', 'vel_cart', 'vel_pole']

    #     policy_tensor = obs['policy']  # tensor([[ 0.2730, -0.2406,  0.0000, -0.0000]], device='cuda:0')

    #     # Extracting the four values
    #     pose_cart = policy_tensor[0, 0].item()  # First value
    #     pose_pole = policy_tensor[0, 1].item()  # Second value
    #     vel_cart = policy_tensor[0, 2].item()   # Third value
    #     vel_pole = policy_tensor[0, 3].item()   # Fourth value
        
    #     # Create a tensor of the values to discretize
    #     obs_tensor = torch.tensor([pose_cart, pose_pole, vel_cart, vel_pole])

    #     # Use torch.bucketize() to find bin indices for each value
    #     state_discretized = tuple(
    #         torch.bucketize(obs_tensor[i], self.bins[i]) for i in range(len(required_keys))
    #     )
        
    #     return state_discretized





    def discretize_state(self, obs: dict):
        """
        Discretize the observation state.

        Args:
        obs (dict): Observation dictionary containing policy states.

        Returns:
            uple[pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]: Discretized state.
        """

        # ========= put your code here =========#
        # cart pose range : [-4.8 , 4.8]
        # pole pose range : [-pi , pi]
        # cart vel  range : [-inf , inf]
        # pole vel range  : [-inf , inf]
        # define number of value
        pose_cart_bin = self.discretize_state_weight[0]
        pose_pole_bin = self.discretize_state_weight[1]
        vel_cart_bin = self.discretize_state_weight[2]
        vel_pole_bin = self.discretize_state_weight[3]
        # pose_cart_bin = 100
        # pose_pole_bin = 720
        # vel_cart_bin = 100
        # vel_pole_bin = 100

        # Clipping value
        pose_cart_bound = 3
        pose_pole_bound = float(np.deg2rad(24.0))
        vel_cart_bound = 15
        vel_pole_bound = 15

        #get observation term from continuos space
        pose_cart_raw, pose_pole_raw , vel_cart_raw , vel_pole_raw = obs['policy'][0, 0] , obs['policy'][0, 1] , obs['policy'][0, 2] , obs['policy'][0, 3]
        pose_cart_clip = torch.clip(pose_cart_raw , -pose_cart_bound ,pose_cart_bound)
        pose_pole_clip = torch.clip(pose_pole_raw , -pose_pole_bound ,pose_pole_bound)
        vel_cart_clip = torch.clip(vel_cart_raw , -vel_cart_bound ,vel_cart_bound)
        vel_pole_clip = torch.clip(vel_pole_raw , -vel_pole_bound ,vel_pole_bound)

        device = pose_cart_clip.device

            # linspace value
        pose_cart_grid = torch.linspace(-pose_cart_bound , pose_cart_bound , pose_cart_bin , device=device)
        pose_pole_grid = torch.linspace(-pose_pole_bound , pose_pole_bound , pose_pole_bin , device=device)
        vel_cart_grid = torch.linspace(-vel_cart_bound , vel_cart_bound , vel_cart_bin , device=device)
        vel_pole_grid = torch.linspace(-vel_pole_bound , vel_pole_bound , vel_pole_bin , device=device)

            # digitalize to range
        pose_cart_dig = torch.bucketize(pose_cart_clip,pose_cart_grid)
        pose_pole_dig = torch.bucketize(pose_pole_clip,pose_pole_grid)
        vel_cart_dig = torch.bucketize(vel_cart_clip,vel_cart_grid)
        vel_pose_dig = torch.bucketize(vel_pole_clip,vel_pole_grid)

        return ( int(pose_cart_dig), int(pose_pole_dig), int(vel_cart_dig),  int(vel_pose_dig))

    # def get_discretize_action(self, obs_dis) -> int:
    #     """
    #     Select an action using an epsilon-greedy policy.

    #     Args:
    #         obs_dis (tuple): Discretized observation.

    #     Returns:
    #         int: Chosen discrete action index.
    #     """
    #     # if self.control_type == ControlType.DOUBLE_Q_LEARNING :
    #     #     if torch.rand(1).item() < self.epsilon:
    #     #         return torch.randint(0, self.num_of_action, (1,)).item()  # Explore
    #     #     else:
    #     #         return int(torch.argmax(self.qa_values[obs_dis]).item() + torch.argmax(self.qb_values[obs_dis]).item()) # Exploit
    #     if self.control_type == ControlType.DOUBLE_Q_LEARNING:
    #         # if torch.rand(1).item() < self.epsilon:
    #         #     return torch.randint(0, self.num_of_action, (1,)).item()  # Explore
    #         # else:
    #         #     # Ensure values are NumPy arrays before conversion
    #         #     qa_numpy = np.array(self.qa_values[obs_dis], dtype=np.float32)
    #         #     qb_numpy = np.array(self.qb_values[obs_dis], dtype=np.float32)

    #         #     # Convert to PyTorch tensors properly
    #         #     qa_tensor = torch.from_numpy(qa_numpy)
    #         #     qb_tensor = torch.from_numpy(qb_numpy)

    #         #     return int(torch.argmax(qa_tensor).item() + torch.argmax(qb_tensor).item())  # Exploit

    #         qa_numpy = np.array(self.qa_values[obs_dis], dtype=np.float32)
    #         qb_numpy = np.array(self.qb_values[obs_dis], dtype=np.float32)

    #         # Convert to PyTorch tensors properly
    #         qa_tensor = torch.from_numpy(qa_numpy)
    #         qb_tensor = torch.from_numpy(qb_numpy)

    #         action_idx = torch.argmax(qa_tensor + qb_tensor).item()
    #         action_idx = max(0, min(action_idx, self.num_of_action - 1))  # Ensure action index is valid
    #         return int(action_idx)

    #     else:
    #         if torch.rand(1).item() < self.epsilon:
    #             return torch.randint(0, self.num_of_action, (1,)).item()  # Explore
    #         else:
    #             return int(torch.argmax(self.q_values[obs_dis]).item())  # Exploit

    #     # else:
    #     #     if torch.rand(1).item() < self.epsilon:
    #     #         return torch.randint(0, self.num_of_action, (1,)).item()  # Explore
    #     #     else:
    #     #         # ตรวจสอบให้แน่ใจว่าเป็น NumPy array ก่อนแปลงเป็น PyTorch tensor
    #     #         q_numpy = np.array(self.q_values[obs_dis], dtype=np.float32)
    #     #         q_tensor = torch.from_numpy(q_numpy)

    #     #         return int(torch.argmax(q_tensor).item())  # Exploit


    def get_discretize_action(self, obs_dis) -> int:
        """
        Select an action using an epsilon-greedy policy for Double Q-Learning.

        Args:
            obs_dis (tuple): Discretized observation.

        Returns:
            int: Chosen discrete action index.
        """
        if self.control_type == ControlType.DOUBLE_Q_LEARNING:
            # Exploration: Choose a random action with probability epsilon
            if torch.rand(1).item() < self.epsilon:
                return torch.randint(0, self.num_of_action, (1,)).item()

            # Exploitation: Use Double Q-Learning selection
            else:
                qa_numpy = np.array(self.qa_values[obs_dis], dtype=np.float32)
                qb_numpy = np.array(self.qb_values[obs_dis], dtype=np.float32)

                # Convert to PyTorch tensors
                qa_tensor = torch.from_numpy(qa_numpy)
                qb_tensor = torch.from_numpy(qb_numpy)

                # Double Q-Learning action selection:
                action_from_q1 = torch.argmax(qa_tensor).item()  # Best action from Q1
                action_from_q2 = torch.argmax(qb_tensor).item()  # Best action from Q2

                # Select the final action based on the maximum Q-value
                best_action = action_from_q1 if qa_tensor[action_from_q1] >= qb_tensor[action_from_q2] else action_from_q2

                # Ensure action index is within valid bounds
                action_idx = max(0, min(best_action, self.num_of_action - 1))
                return int(action_idx)

        else:
            # Standard Q-learning with epsilon-greedy policy
            if torch.rand(1).item() < self.epsilon:
                return torch.randint(0, self.num_of_action, (1,)).item()  # Explore
            else:
                self.q_values[obs_dis] = torch.tensor(self.q_values[obs_dis], dtype=torch.float32)
                return int(torch.argmax(self.q_values[obs_dis]).item())  # Exploit


    def mapping_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value.

        Args:
            action (int): Discrete action in range [0, n]

        Returns:
            torch.Tensor: Scaled action tensor.
        """
        action_min, action_max = self.action_range
        action_continuous = action_min + (action / (self.num_of_action - 1)) * (action_max - action_min)
        return torch.tensor(action_continuous, dtype=torch.float32)

    # def get_action(self, obs) -> torch.Tensor:
    #     """
    #     Get action based on epsilon-greedy policy.

    #     Args:
    #         obs (dict): The observation state.

    #     Returns:
    #         torch.Tensor, int: Scaled action tensor and chosen action index.
    #     """
    #     obs_dis = self.discretize_state(obs)
    #     # action_idx = torch.tensor(self.get_discretize_action(obs_dis), dtype=torch.int)
    #     action_idx = torch.tensor([self.get_discretize_action(obs_dis)], dtype=torch.int)
    
    #     action_value = self.mapping_action(action_idx.item())

    #     # if isinstance(action_value, torch.Tensor):
    #     #     action_tensor = action_value.view(1, 1)
    #     # else:
    #     #     
    #     action_tensor = torch.tensor([[action_value]])

    #     return action_tensor, action_idx  

    def get_action(self, obs) -> torch.Tensor:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (dict): The observation state.

        Returns:
            torch.Tensor, int: Scaled action tensor and chosen action index.
        """
        obs_dis = self.discretize_state(obs)
        # action_idx = torch.tensor(self.get_discretize_action(obs_dis), dtype=torch.int)
        action_idx = self.get_discretize_action(obs_dis)
        action_tensor = self.mapping_action(action_idx)
    


        # if isinstance(action_value, torch.Tensor):
        #     action_tensor = action_value.view(1, 1)
        # else:
        #     
        # action_tensor = torch.tensor([[action_value]])

        return action_tensor.unsqueeze(0).unsqueeze(0), action_idx  
    
    def decay_epsilon(self, total_episodes ):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # self.epsilon = max(self.final_epsilon, self.epsilon * (self.epsilon_decay ** episode))
        # self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

        # if total_episodes >
        epsilon_decrease = (1.0 - self.final_epsilon) / total_episodes # Calculate how much to decrease each step
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decrease)

    def save_q_value(self, path, filename):
        """
        Save the model parameters to a JSON file.

        Args:
            path (str): Path to save the model.
            filename (str): Name of the file.
        """
        # Convert tuple keys to strings
        try:
            q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
        except:
            q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
        if self.control_type == ControlType.MONTE_CARLO:
            try:
                n_values_str_keys = {str(k): v.tolist() for k, v in self.n_values.items()}
            except:
                n_values_str_keys = {str(k): v for k, v in self.n_values.items()}
        
        # Save model parameters to a JSON file
        if self.control_type == ControlType.MONTE_CARLO:
            model_params = {
                'q_values': q_values_str_keys,
                'n_values': n_values_str_keys
            }
        else:
            model_params = {
                'q_values': q_values_str_keys,
            }
        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(model_params, f)

            
    def load_model(self, path, filename):
        """
        Load model parameters from a JSON file.

        Args:
            path (str): Path where the model is stored.
            filename (str): Name of the file.

        Returns:
            dict: The loaded Q-values.
        """
        full_path = os.path.join(path, filename)        
        with open(full_path, 'r') as file:
            data = json.load(file)
            data_q_values = data['q_values']
            for state, action_values in data_q_values.items():
                state = state.replace('(', '')
                state = state.replace(')', '')
                tuple_state = tuple(map(float, state.split(', ')))
                self.q_values[tuple_state] = action_values.copy()
                if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                    self.qa_values[tuple_state] = action_values.copy()
                    self.qb_values[tuple_state] = action_values.copy()
            if self.control_type == ControlType.MONTE_CARLO:
                data_n_values = data['n_values']
                for state, n_values in data_n_values.items():
                    state = state.replace('(', '')
                    state = state.replace(')', '')
                    tuple_state = tuple(map(float, state.split(', ')))
                    self.n_values[tuple_state] = n_values.copy()
            return self.q_values

