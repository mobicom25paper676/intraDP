mode=${1-"flex"}    # flex means no latency constraint; fix means 1 hz latency constraint
cmd=${2-"rosrun kapao pose_follower.py"}
env=${3-"indoors"}
container=${4-"robot2"}
dur=${5-1800}
workload=${6-"kapao"}
offload=${7-"True"}
username=${8-"user"}
ip=${9-"192.168.50.11"}
port=${10-"12345"}
log_dir="${mode}_${env}_${workload}"
bag_name=${11-pose_follower_offload.bag}
echo "offload_mode: $mode; env: $env; cmd: $cmd; offload: $offload"

echo "log_dir $work/log/$log_dir"

tmux has-session -t offload_exp 2>/dev/null
if [ $? != 0 ]; then
  # Set up your session
    tmux new -t offload_exp -d

    sleep 5
    tmux new-window -t offload_exp -n work
    tmux new-window -t offload_exp -n roscore
    tmux new-window -t offload_exp -n rosbag
    sleep 3
fi

tmux send-keys -t offload_exp:roscore C-c ENTER; tmux send-keys -t offload_exp:roscore "rosclean purge -y" ENTER; 
tmux send-keys -t offload_exp:rosbag C-c ENTER
tmux send-keys -t offload_exp:roscore "roscore" ENTER
sleep 5
# tmux send-keys -t offload_exp:rosbag "rosparam set /use_sim_time true; cd; rosbag play -l --clock all_topic.bag" ENTER

ssh $username@$ip "tmux has-session -t replay_bw; if [ \$? != 0 ]; then tmux new -t replay_bw -d; sleep 3; tmux new-window -t replay_bw -n server; sleep 2; fi"
sleep 10
ssh $username@$ip "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server C-c ENTER;  tmux send-keys -t replay_bw:server \"cd \\\$work; mkdir -p \\\$work/log/$log_dir; python3.8 start_server.py 0.0.0.0 12345 \\\$work/exp_utils/$env.txt &> \\\$work/log/$log_dir/server.log\" ENTER;"

echo "Started server"
# tmux send-keys -t offload_exp:env "docker exec -it $container zsh" ENTER
# tmux send-keys -t offload_exp:env "roslaunch turn_on_wheeltec_robot ros_torch_env.launch" ENTER
# echo "Started wheeltech robot sensors and motion controller"

tmux send-keys -t offload_exp:work "rm -rf \$work/log/$log_dir; mkdir -p \$work/log/$log_dir; rosparam set /offload_mode $mode; rosparam set /server_ip $ip; rosparam set /server_port $port; rosparam set /offload_method $mode; rosparam set /offload_port $port; rosparam set /offload $offload" ENTER
# nice -n [priority] cmd
tmux send-keys -t offload_exp:work "ROS_LOG_DIR=\$work/log/$log_dir $cmd" ENTER
echo "Running $cmd"

echo "Worload started. Killing in ${dur}s."

sleep 4
sleep $dur

echo "Killing everything..."
tmux send-keys -t offload_exp:rosbag C-c ENTER
#rosnode kill -a
# tmux send-keys -t offload_exp:env C-c ENTER

tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
ssh $username@$ip "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server C-c ENTER;"
ssh $username@$ip "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server C-c ENTER;"
ssh $username@$ip "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server C-c ENTER;"
sleep 5
# tmux send-keys -t offload_exp:bw C-c ENTER
pkill -9 -f "python3 $work*"
pkill -9 -f "python3 $work*"
pkill -9 -f "python3 $work*"
pkill -9 -f "python3 $work*"
pkill -9 -f "python3 $work*"
pkill -9 -f "python3 $work*"
pkill -9 -f "python3 $work*"
pkill -9 -f "python3 $work*"
sleep 5

echo "Fin"
echo ""

