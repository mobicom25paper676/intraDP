models=(MobileNet_V3_Small ResNet101 VGG19_BN)
envs=(indoors outdoors)
dur=${1-300}

for env in ${envs[*]}; do
    for model in ${models[*]}; do
        bash start_work.sh tp "python3 /project/ParallelCollaborativeInference/ros_ws/src/torchvision/scripts/run_torchvision.py  -a $model -d ImageNet -p tp" $env robot2_torch13 $dur torchvision_$model
    done
done
