envs=(indoors outdoors)
# offload_methods=(all fix flex mixed2)
offload_methods=(mixed2)
# offload_methods=(local all fix flex mixed2)
# datasets=(CIFAR10 CIFAR10 OxfordIIITPet OxfordIIITPet OxfordIIITPet OxfordIIITPet)
# tasks=(classification classification segmentation segmentation detection detection)
models=(VGG19_BN DenseNet121 ConvNeXt_Large ConvNeXt_Base RegNet_X_16GF)
dur=600
user=user
ip=192.168.50.11
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
        echo "Running $env $method kapao cases..."
        if [ $method == "local" ]
        then
            bash start_work.sh $method "rosrun kapao pose_follower.py" $env robot2_torch13 $dur kapao False

            bash start_work.sh $method "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav False
        else
            bash start_work.sh $method "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav True  $user $ip $port
        fi 

    done
done




