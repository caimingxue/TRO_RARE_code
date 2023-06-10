# RARE

## Description:
The TRO_REMA is an object-oriented Python implementation of a magnetic application. The package includes an automated collection of coil fields for calibration and a simulation environment for magnetic robots. 
The package automated field data collection is used to collect coil field data using a robotic arm and a magnetic sensor.
The package simulation_environment is used to model the dynamic robot and medical scenario via the ROS RVIZ. The magnetic sensor: TLE493D W2B6, Infineon




## Installation
This requires Ubuntu 18.04 with Python 3. Meanwhile, ROS melodic is required for the communication framework. 

#### Prerequisites

- **Operating System**: tested on Ubuntu 18.04.
- **ROS**: Melodic.
- **Python Version**: >= 3.8.0.



#### Install Dependencies

You can either install the dependencies in a conda virtual env (recomended) or manually. 

For conda virtual env installation, simply create a virtual env named **RARE** by:

```
conda env create -f environment.yml
```

If you prefer to install all the dependencies by yourself, you could open `environment.yml` in editor to see which packages need to be installed by `pip`.



## Usage automated field data collection

Run automated field data collection using shell commands 
Open a terminal and go to the / automated field data collection directory 
Python field_read_port.py (read the real-time coil data using the magnetic sensor)
Python robotic_arm_control.py (control the robotic arm to travel the designed grids)


## Usage simulation_environment
To construct the magnetic control simulation environment follow the steps below:
Step1: Use Solidworks to make the magnetic robot STL file. Then put the STL file to the \simulation_environment\STL_files.
Step2: Run python simulation_environment_rviz.py.


if you have any questions, please contact: mingxuecai@outlook.com






