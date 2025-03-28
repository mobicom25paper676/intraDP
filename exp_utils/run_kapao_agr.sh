envs=(indoors outdoors)
# offload_methods=(local all flex fix)
offload_methods=(fix flex all)
# offload_methods=(local all fix flex mixed2)
# datasets=(CIFAR10 CIFAR10 OxfordIIITPet OxfordIIITPet OxfordIIITPet OxfordIIITPet)
# tasks=(classification classification segmentation segmentation detection detection)
models=(DenseNet121 VGG19_BN ConvNeXt_Large ConvNeXt_Base RegNet_X_16GF)
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

        echo ""
        echo "Running $env $method kapao cases..."
        if [ $method == "local" ]
        then
            bash start_work.sh $method "python3 \$work/kapao_test.py" $env robot2_torch13 300 kapao True $user $ip $port

            bash start_work.sh $method "python3 \$work/ros_ws/src/agrnav/scripts/inference_ros.py" $env robot2_torch13 300 agrnav True $user $ip $port
        else
            bash start_work.sh $method "python3 \$work/kapao_test.py" $env robot2_torch13 $dur kapao True  $user $ip $port
            
            bash start_work.sh $method "python3 \$work/ros_ws/src/agrnav/scripts/inference_ros.py" $env robot2_torch13 $dur agrnav True  $user $ip $port
        fi 

    done
done




