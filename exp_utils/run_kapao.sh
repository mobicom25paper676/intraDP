modes=(mixed2 fix flex)
envs=(indoors outdoors)
dur=${1-1800}

for env in ${envs[*]}; do
    for mode in ${modes[*]}; do
        bash start_work.sh $mode "rosrun kapao pose_follower.py" $env robot2_torch13 $dur kapao
    done
done
echo "Running local cases..."
bash start_work.sh flex "rosrun kapao pose_follower.py" indoors robot2_torch13 300 kapao_local False

