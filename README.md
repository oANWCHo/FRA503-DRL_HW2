# FRA503 Deep Reinforcement Learning for Robotics

# Instruction

## Recommend using [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)

Download Miniconda different version, IsaacLab using python version 3.10 [[list of Miniconda](https://repo.anaconda.com/miniconda)].

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh
```

Install Miniconda

```
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

The installer finishes and displays, “Thank you for installing Miniconda3!”

Close and re-open your terminal window for the installation to fully take effect, or use the following command to refresh the terminal

```
source ~/.bashrc
```

### Verifying the Miniconda installation

Test your installation by running `conda list`. If conda has been installed correctly, a list of installed packages appears.

![alt text](image-1.png)

If you see this, then the installation was successful! 🎉

## Installing Isaac Sim & Isaac Lab

### Pip Installation (recommended for Ubuntu 22.04)

Follow the Installing and Verifying steps [[link](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)]

### Binary Installation (recommended for Ubuntu 20.04)

Follow the Installing and Verifying steps [[link](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)]

### Verifying the Isaac Lab installation

```
# Option 1: Using the isaaclab.sh executable
# note: this works for both the bundled python and the virtual environment
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

# Option 2: Using python in your virtual environment
python source/standalone/tutorials/00_sim/create_empty.py
```

![alt text](image.png)

If you see this, then the installation was successful! 🎉


## Isaac Lab Overview [Optional]

For more understanding of IsaacLab you can go through Isaac Lab Overview.

1. Developer’s Guide [[link](https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/index.html)]
2. Core Concepts [[link](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/index.html)]
3. Sensors [[link](https://isaac-sim.github.io/IsaacLab/main/source/overview/sensors/index.html)]

## Available Environments

The following lists comprises of all the RL tasks implementations that are available in Isaac Lab. [[link](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)]

or

you can excute following command line

```
python source/standalone/environments/list_envs.py
```

![alt text](image-2.png)

## Tutorials

We recommend that you go through the tutorials in the order they are listed here.

### Simulation Overview 

1. Setting up a Simple Simulation [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html#setting-up-a-simple-simulation)]
2. Interacting with Assets [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html#interacting-with-assets)]
3. Creating a Scene [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html#creating-a-scene)]

### Task Design Workflows

For more detail of different workflows for designing environments. [[link](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html)]

4. Designing an Environment [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html#designing-an-environment)]

    `HW0 Requirement`: You need to understanding `Creating a Manager-Based Base Environment` and `Creating a Manager-Based RL Environment` in designing an environment.

    4.1 `Creating a Manager-Based Base Environment` [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html)]

    4.2 `Creating a Manager-Based RL Environment` [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)]

### [Optional]

5. Integrating Sensors [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html#integrating-sensors)]

6. Using motion generators [[link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html#using-motion-generators)]

## How-to Guides [Optional]

This section includes guides that help you use Isaac Lab. 
These are intended for users who have already worked through the tutorials and are looking for more information on how to use Isaac Lab. 
If you are new to Isaac Lab, we recommend you start with the tutorials. [[link](https://isaac-sim.github.io/IsaacLab/main/source/how-to/index.html#how-to-guides)]