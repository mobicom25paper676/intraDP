envs=(indoors outdoors)
# offload_methods=(local all flex fix)
offload_methods=(fix flex all local)
# offload_methods=(local all fix flex mixed2)
# datasets=(CIFAR10 CIFAR10 OxfordIIITPet OxfordIIITPet OxfordIIITPet OxfordIIITPet)
# tasks=(classification classification segmentation segmentation detection detection)
models=(ResNet152)
dur=1200
user=user
ip=192.168.50.132
port=12345
for method in ${offload_methods[*]}; do
    for env in ${envs[*]}; do

        if [ $method == "local" ] && [ $env == "outdoors" ]
        then
            echo ""
            echo "Skipping $env $method cases..."
            continue
        fi

        echo ""
        echo "Running $env $method torchvision cases..."
        # if [ $method == "all "]     # only need to add all for torchvision
        # then
        for i in {0..0}; do  # TODO run all
            if [ $method == "local" ]
            then
                bash start_work.sh $method "python3 \$work/ros_ws/src/torchvision/scripts/run_torchvision.py -a ${models[i]} -d CIFAR10" $env robot2_torch13 300 ${models[i]} True $user $ip $port
            else
                bash start_work.sh $method "python3 \$work/ros_ws/src/torchvision/scripts/run_torchvision.py -a ${models[i]} -d CIFAR10" $env robot2_torch13 $dur ${models[i]} True $user $ip $port
            fi
        done

    done
done




