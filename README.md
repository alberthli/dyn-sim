# Dynamical Simulation Environment

## TODOs
- [] Controllers
	- [] General API for constraints (pick optimizer)
	- [] MPC via successive linearization
	- [] iLQR
- [] Simulator
	- [X] Add back in animation
	- [] Add back in obstacles
	- [] Clean up disturbance injection API once everything else is stable
- [] (Long-Term) Observers

## Overview
This repository is a catch-all simulation environment for studying controllers (which may have bandwidth limitations) for dynamical systems.
### File Structure
* `dyn_sim` - core code
	* `ctrl` - implemented controllers with subfolders corresponding to each system. General control code lives in the parent folder (e.g. `mpc`).
		* `quad`
			* Inner/Outer Loop PD Controller
			* [REFACTORING] Safe Multirate Controller
		* [WIP] `segway`
	* `sim` - simulator code
	* `dyn_sys` - implemented dynamical systems
		* `planar` - systems that are visualized in the plane
			* [WIP] segway
		* `spatial` - systems that are visualized in Euclidean space
			* quad
	* `util` - common utility functions across codebase
* `scripts` - scripts for running simulations

## Dependencies and Environment Management
### Gurobi
This repo uses the optimization package Gurobi, which is a commercial product (but is available to academics for free). This section details (hopefully) headache-free installation of Gurobi for academics. If you are not an academic, wait for alternate free optimizer support (e.g. `scipy`).

First, register using your academic credentials on the [website](https://www.gurobi.com/academia/academic-program-and-licenses/#:~:text=After%20registering%20and%20logging%20in,Follow%20the%20instructions%20in%20README) (instructions for the academic license are linked here).

Second, we will only require the package `gurobipy`, so we will install that with the requirements file in the next section. However, we must license our installation, so install the license tools package [here](https://support.gurobi.com/hc/en-us/articles/360059842732). Extract the executable file `grbgetkey` into any directory.

Third, on the Gurobi website, you should be able to [view your current licenses](https://www.gurobi.com/downloads/licenses/). If you click on your License ID, you will come to a page with Installation instructions, which should include a line that looks like:
```
grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```
Copy this command, and in your terminal, run the command
```
<full_parent_path>/grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```
with the correct parent path and license key. Save the license file somewhere logical, such as the home directory. `gurobipy` should be usable on your machine after its installation via `pip`.

### Dependency Installation
This repo is designed to be run in a `conda` virtual environment with dependencies managed by `pip` and versions frozen in a requirements file (versions last frozen May 23, 2022).

To make an environment, make sure you have conda installed and then run the command:
```
conda create --name <env_name> python=3.10.4
```
Anytime after initializing the environment, activate before using project files:
```
conda activate <env_name>
```
To install the dependencies in the project, after activating the environment for the first time, run
```
pip install -r requirements.txt
```

### Development
For people developing on the environment, style-checking and static type-checking is performed using pre-commit hooks. Make sure you have `pre-commit` installed (separate command from dependency installation):
```
pip install pre-commit
pre-commit install
```
