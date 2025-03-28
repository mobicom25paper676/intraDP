# on server
docker build -f Dockerfile.server -t inference .
docker run -v $PWD:/workspace -itd --gpus all --net host --ipc host --name inference inference
docker attach inference

# on robot
docker build -f Dockerfile.robot -t inference .
docker run -v $PWD:/workspace -itd --runtime nvidia --net host --ipc host --name inference inference
docker attach inference