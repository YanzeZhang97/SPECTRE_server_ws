This is the server side implementation of the SPECTRE for ITSC 6166/8166 Computer Communivation and Networks. 

The server side would be a workstation it will be used for deploy the trained model. The code is tested in the `ROS2 foxy` envrionment.

## Configuration
Similar to the original project, you need to install all the required libaries/packages. The suggested way is to using a conda virtual environment. You can follow the steps to make the service runing:

**step 1** Create a virtual env

`Conda create -n spectre8 python=3.8`

**step 2** Make a workspace, clone the code and install the requirements

`mkdir -p ros2_ws`

`cd ros2_ws`

`git clone https://github.com/YanzeZhang97/SPECTRE_server_ws.git`



**step 3** Install additional requirements

`pip install -r requirement.txt`