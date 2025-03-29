# Intra-DP: A High Performance Collaborative Inference System for Mobile Edge Computing

Intra-DP is a high-performance distributed inference system optimized for DNN inference on mobile edge computing. 
Intra-DP employs a novel parallel computing technique based on local operators (i.e., operators whose minimum unit input is not the entire input tensor, such as the convolution kernel). 
By cutting their operations into several independent sub-operations and overlapping the computation and transmission of different sub-operations through parallel execution, Intra-DP significantly mitigates transmission bottlenecks in mobile edge computing, achieving fast and energyefficient inference.

We put Intra-DP to the test on our real-world robot, evaluating its performance in two typical real-world robotic applications: [KAPAO](https://github.com/wmcnally/kapao) and [AGRNav](https://github.com/jmwang0117/AGRNav) (as shown in `kapao-demo.mp4` and `agrnav-demo.mp4`). Additionally, we assessed Intra-DP's effectiveness on several models implemented in Torchvision (you can find these in `./ros_ws/src/torchvision/scripts/run_torchvision.py`).

To make it easier for you to explore and understand our work, we've organized our codebase as follows:
- The source code for Intra-DP is located in the `intraDP` folder.
- The scripts we used in our experiments can be found in the `exp_utils` folder.
- The relevant code for our workloads is placed in the `ros_ws` folder and two submodules of `KAPAO` and `AGRNav` in `third_parties`.
- We've also included some examples of our experiment logs in the `log` folder, giving you a glimpse into our experiment process.

We hope this structure helps you navigate our project more efficiently. If you have any questions or need further clarification on any aspect of our work, please don't hesitate to reach out. We're always excited to discuss our research and hear your thoughts and feedback!


## Installation
1. Clone this repo and enter the project folder.

2. Building and initiating the corresponding docker containers on both the server and robot sides based on the dockerfile file we provided (`Dockerfile.robot`, `Dockerfile.ros_robot` and `Dockerfile.server`), or execute the script we provided directly.

```
bash run_docker.sh
```
Note that `Dockerfile.ros_robot` is a special version of dockerfile for our robot hardware, as KAPAO and AGRNav needs to control the movement of the robot via ROS.

3. Install the dependency packages and intraDP in the docker containers on both the server and robot sides:
```
pip install -r requirement.txt
python setup.py install
```


## How to Use?
1. Integrate Intra-DP into your existing applications, requiring only three lines of code. 
For instance, applying Intra-DP to a VGG19 model is shown as follows, where ``192.168.50.1'' is the IP address of the GPU server.

```python
# Import package of Intra-DP
import intraDP
# Define a VGG19 model as usual
vgg19 = VGG19().to(device)
# Apply Intra-DP
IDP = intraDP(ip = "192.168.50.1")
IDP.start_client(model = vgg19)
# Run model for inference as usual
result = vgg19(input)
```
The corresponding modifications to our workload's source code, which are conveniently provided in the `ros_ws` folder.

2. To get started, simply run the following script on your GPU server to launch the Intra-DP server:
```
python exp_utils/start_server.py #on GPU server
```

It's essential to ensure that the Intra-DP client and server can communicate seamlessly, so please double-check the parameters on the GPU server side before running your application.

3. Once the Intra-DP server is up and running, you can start model inference on your mobile devices as usual via the Intra-DP client.

```
python ./ros_ws/src/torchvision/scripts/run_torchvision.py #on robot
```

Alternatively, we've provided a collection of scripts in the `exp_utils` folder specifically designed to streamline the execution of our workloads. Feel free to explore and utilize these scripts to simplify your experimentation process.


## Cite Us
Upcoming, the paper is under review.
