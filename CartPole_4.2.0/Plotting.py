import json
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def plot_pose_cart_and_pole(q_values, discretize_state_weight, pose_cart_bound, pose_pole_bound):
    # Create meshgrid for pose_cart and pose_pole
    pose_cart = np.linspace(pose_cart_bound[0], pose_cart_bound[1], discretize_state_weight[0])
    pose_pole = np.linspace(pose_pole_bound[0], pose_pole_bound[1], discretize_state_weight[1])
    pose_cart_grid, pose_pole_grid = np.meshgrid(pose_cart, pose_pole)

    # Initialize a Q-value grid for plotting
    q_value_grid = np.zeros(pose_cart_grid.shape)

    # Populate the Q-value grid with data from the Q-values
    for key, q_vals in q_values.items():
        state = eval(key)  # Convert the string tuple into a tuple
        pose_cart_idx, pose_pole_idx, _ , _ = state  # We assume the other indices are not needed for the plot
        q_value_grid[pose_pole_idx, pose_cart_idx] = max(q_vals)  # Assign the max Q-value for each state-action pair

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(pose_cart_grid, pose_pole_grid, q_value_grid, cmap='viridis')

    ax.set_xlabel('Pose Cart')
    ax.set_ylabel('Pose Pole')
    ax.set_zlabel('Q-value')
    ax.set_title('3D Surface Plot of Q-values')

    ax.set_zlim(0, 0.1)
    ax.set_xticks(np.linspace(pose_cart_bound[0], pose_cart_bound[1], discretize_state_weight[0]))  # X-axis: pose_cart
    ax.set_yticks(np.linspace(pose_pole_bound[0], pose_pole_bound[1], discretize_state_weight[1]))  # Y-axis: pose_pole

    plt.show()

def plot_pole_and_vel_cart(q_values, discretize_state_weight, pose_pole_bound, vel_cart_bound):
    # Create meshgrid for pose_pole and vel_cart
    pose_pole = np.linspace(pose_pole_bound[0], pose_pole_bound[1], discretize_state_weight[1])
    vel_cart = np.linspace(vel_cart_bound[0], vel_cart_bound[1], discretize_state_weight[2])
    # print(pose_pole, vel_cart)
    pose_pole_grid, vel_cart_grid = np.meshgrid(pose_pole, vel_cart)

    # print(pose_pole_grid,vel_cart_grid)


    # Initialize a Q-value grid for plotting
    q_value_grid = np.zeros((discretize_state_weight[1], discretize_state_weight[2]))

    # print(q_value_grid)

    # Populate the Q-value grid with data from the Q-values
    for key, q_vals in q_values.items():
        state = eval(key)  # Convert the string tuple into a tuple
        _, pose_pole_idx, vel_cart_idx, _ = state  # We assume the other indices are not needed for the plot

        # Debug: Print the indices to check if they're within the correct range
        # print(f"State: {state}, pose_pole_idx: {pose_pole_idx}, vel_cart_idx: {vel_cart_idx}")

        # Map indices within bounds (use modulus to wrap them within bounds)
        pose_pole_idx = pose_pole_idx % discretize_state_weight[1]  # Map to [0, discretize_state_weight[1] - 1]
        vel_cart_idx = vel_cart_idx % discretize_state_weight[2]  # Map to [0, discretize_state_weight[2] - 1]

        # print(f"Mapping pose_pole_idx: {pose_pole_idx}, vel_cart_idx: {vel_cart_idx}")

        # Assign the max Q-value for each state-action pair
        q_value_grid[pose_pole_idx, vel_cart_idx] = max(q_vals)

    q_value_grid = q_value_grid.T 
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(pose_pole_grid, vel_cart_grid, q_value_grid, cmap='viridis')

    ax.set_xlabel('Pose Pole')
    ax.set_ylabel('Velocity Cart')
    ax.set_zlabel('Q-value')
    ax.set_title('3D Surface Plot of Q-values (Pose Pole vs Velocity Cart)')
    # ax.set_title(Algorithm_name,episode,discount,learning_rate)

    ax.set_zlim(0, 0.1)
    ax.set_xticks(np.linspace(pose_pole_bound[0], pose_pole_bound[1], discretize_state_weight[1]))  # X-axis: pose_pole
    ax.set_yticks(np.linspace(vel_cart_bound[0], vel_cart_bound[1], discretize_state_weight[2]))  # Y-axis: vel_cart

    plt.show()




# Example usage
# Define the discretization weights, bounds for pose_cart, pose_pole, and vel_cart
pose_cart_bound = [-3, 3]
pose_pole_bound = [-24, 24]
vel_cart_bound = [-15, 15]

# hyperparameters
num_of_action = 11
action_range = [-16.0, 16.0]  # [min, max]
discretize_state_weight = [5, 11, 3, 3]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
learning_rate = 0.3
start_epsilon = 0
epsilon_decay = 0 # reduce the exploration over time
final_epsilon = 0
discount = 0.5
    
task_name =  "Stabilize" # Stabilize, SwingUp
Algorithm_name = "Double_Q_Learning"  

episode = 10800 # Edit here to match an episode on the json file
    
q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}_{discount}_{learning_rate}.json"
print(q_value_file) # Verify that the correct json

full_path = os.path.join(f"q_value/{task_name}", Algorithm_name)
json_file_path  = full_path + "/" + q_value_file

# Load Q-values from JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)
    q_values = data["q_values"]

# Call the function with your file path
# plot_pose_cart_and_pole(q_values, discretize_state_weight, pose_cart_bound, pose_pole_bound)
plot_pole_and_vel_cart(q_values, discretize_state_weight, pose_pole_bound, vel_cart_bound)
