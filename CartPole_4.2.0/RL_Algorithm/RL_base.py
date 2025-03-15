import torch
from collections import defaultdict
from enum import Enum


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
        self.bins = [
            torch.linspace(-2.4, 2.4, discretize_state_weight[0]),  # pose_cart bins
            torch.linspace(-0.209, 0.209, discretize_state_weight[1]),  # pose_pole bins
            torch.linspace(-2, 2, discretize_state_weight[2]),  # vel_cart bins
            torch.linspace(-2, 2, discretize_state_weight[3])  # vel_pole bins
        ]

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

    def discretize_state(self, obs: dict):
        """
        Discretize the observation state using predefined bins.

        Args:
            obs (dict): Observation dictionary.

        Returns:
            Tuple[int, int, int, int]: Discretized state.
        """
        required_keys = ['pose_cart', 'pose_pole', 'vel_cart', 'vel_pole']
        obs_tensor = torch.tensor([obs[key] for key in required_keys])

        # Use torch.bucketize() to find bin indices
        state_discretized = tuple(
            torch.bucketize(obs_tensor[i], self.bins[i]) for i in range(len(required_keys))
        )
        return state_discretized

    def get_discretize_action(self, obs_dis) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            obs_dis (tuple): Discretized observation.

        Returns:
            int: Chosen discrete action index.
        """
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.num_of_action, (1,)).item()  # Explore
        else:
            return torch.argmax(self.q_values[obs_dis]).item()  # Exploit

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

    def get_action(self, obs) -> torch.Tensor:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (dict): The observation state.

        Returns:
            torch.Tensor, int: Scaled action tensor and chosen action index.
        """
        obs_dis = self.discretize_state(obs)
        action_idx = torch.tensor(self.get_discretize_action(obs_dis), dtype=torch.int)

        action_scalar = action_idx.item() if action_idx.numel() == 1 else action_idx
        action_value = self.mapping_action(action_scalar)

        if isinstance(action_value, torch.Tensor):
            action_tensor = action_value.view(1, 1)
        else:
            action_tensor = torch.tensor([[action_value]], dtype=torch.float32)

        return action_tensor, action_idx  

    def decay_epsilon(self, episode):
        """
        Decay epsilon value to reduce exploration over time.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon * (self.epsilon_decay ** episode))

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

            
    def load_q_value(self, path, filename):
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

