# Dynamical Simulation Environment

## Overview
This repository is a catch-all simulation environment for studying controllers (which may have bandwidth limitations) for dynamical systems.
### File Structure
* `dyn_sim` - core code
	* `ctrl` - implemented controllers with subfolders corresponding to each system
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

## Environment Management
### Dependency Installation
This repo is designed to be run in a `conda` virtual environment with dependencies managed by `pip` and versions frozen in a requirements file (versions last frozen May 14, 2022). To make an environment, make sure you have conda installed and then run the command:
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
