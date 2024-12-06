**This is the server side implementation of the SPECTRE for ITSC 6166/8166 Computer Communivation and Networks.**

**This is another implementation for the [original project](https://github.com/ibrahim-anas/SPECTRE-ROS2). In this implementation, you are able to run a service in the [workstation side](https://github.com/YanzeZhang97/SPECTRE_server_ws) and run a image collection in the [robot/client side](https://github.com/YanzeZhang97/SPECTRE_client_ws).**

The server side would be a workstation it will be used for deploy the trained model. The code is tested in the `ROS2 foxy` envrionment. Please make sure you have `ROS2 foxy` installed.

## Configuration
Similar to the original project, you need to install all the required libaries/packages. The suggested way is to using a conda virtual environment. You can follow the steps to make the service runing:

**step 1** Create a virtual env

`Conda create -n spectre8 python=3.8`

**step 2** Make a workspace, clone the code and install the requirements

`mkdir -p ros2_ws`

`cd ros2_ws`

`git clone https://github.com/YanzeZhang97/SPECTRE_server_ws.git`

`pip install -r requirement.txt`

**step 3** Install additional requirements

`conda install pytorch3d -c pytorch3d`

`cd python_service/python_service/external/face_alignment`

`pip install -e .`

`cd ../face_alignment`

`git lfs pull`

`pip install -e .`

**step 4** Download the pretrained model

`cd ../../../../`

`bash quick_install.sh`

## Running the code

`colcon build`

`source install/setup.bash`

`ros2 run python_service pythonservice`

To see the outputs, please run

`ros2 run rqt_image_view rqt_image_view`
