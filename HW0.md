# Homework 0

After you have done following instruction let continue on to HW part.

For the first homework, we will look at an ideal RL example: training an agent to solve the classic Cartpole control problem. The objective of this homework is to provide an overview of RL components and the basic concept of how each component is relate to each other.

### Learning Objectives:

1.  Understand the RL training pipeline in IsaacLab:

    - Environment & Agent Setup
    - Training process
    - Performance evaluation

        - Viewing learning performance logs
        - Viewing trained agent play in environment

2. Understand the IsaacLab workflow `Manager-Based RL Environment` for implementation your RL project setup in IsaacLab:

    - Action
    - Observation
    - Event
    - Reward
    - Termination

3. Connect practical experience with theoretical RL knowledge.

### Part 1: Playing with `Cartpole` RL Agent [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html#training-with-an-rl-agent)]

You can try to changing parameters in following paths: `omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg.py` 

Possible modifications
- Action
- Observation
- Event
- Reward
- Termination

Train the RL agent, compare experimental results and write a report.

`Submitted:` compare experimental report.

`Hint:` compare acheived reward using tensorboard logs.

### Part 2: Exploring additional available enviroments [Optional] [[link](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html#available-environments)]

`Submitted:` compare experimental report.

### Part 3: Mapping RL Fundamentals

Answer questions relating to RL concepts learned in class using the provided diagram.

- What is reinforcement learning and its components according to your understanding? Giving examples of each component consider the `Cartpole` problem.

- What is the difference between the reward, return, and the value function?

- Consider policy, state, value function, and model as mathematical functions, what would each one take as input and output? 

- According to the last question, if we considered the problems that held the Markov property, what will be changed? How are the changed benefit to our learning framework?



