import argparse
import pickle
import os
import os.path as osp
import time
from inspect import getmembers, isclass

import sys
project_dir = osp.abspath(osp.join(*([osp.abspath(__file__)] + [os.pardir] * 5)))
sys.path.insert(0, project_dir)
from threading import Event
import rospy
from std_srvs.srv import Empty, EmptyResponse

import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models, datasets

tasks = ["all", "classification", "detection", "segmentation", "video"]
parser = argparse.ArgumentParser(description='torch vision inference')
parser.add_argument('-t', '--task', default='classification',
                    choices=tasks,
                    help='task: ' +
                    ' | '.join(tasks) +
                    ' (default: classification)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                    help='model architecture')
parser.add_argument('-d', '--dataset', default='CIFAR10',
                    help='dataset')
parser.add_argument('-p', '--parallel', default='select')
parser.add_argument('--no-offload', action='store_true',
                    help='no offload')

def get_model_weights(model_arch: str, module):
    all_class = getmembers(module, isclass)
    model_arch = model_arch.upper()
    cls = None
    for cls_name, cls in all_class:
        if cls_name.upper().startswith(model_arch):
            break
    assert cls is not None, f"{model_arch} not found."
    return cls

def start_service(_):
    start.set()
    return EmptyResponse()

if __name__ == "__main__":
    rospy.init_node("run_torchvision")
    args = parser.parse_args()
    task = args.task
    model_arch: str = args.arch
    parallel_approach: str = args.parallel
    # if parallel_approach == "tp":
    #     assert model_arch.lower().startswith("vgg"),\
    #         f"Unsupported model arch {model_arch} with parallel approach {parallel_approach}"
    model_arch = model_arch.lower()
    dataset_name: str = args.dataset

    data_dir = osp.join(os.environ["work"], "data", dataset_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    rospy.loginfo(f"Using data dir: {data_dir};")
    rospy.loginfo(f"device {device}; task {task}; model_arch {model_arch}; dataset {dataset_name}")
    if task != "classification":
        models = getattr(models, task)

    weights = get_model_weights(model_arch, models).DEFAULT
    model: torch.nn.Module = getattr(models, model_arch)(weights=weights)
    preprocess = weights.transforms()
    model.eval()
    model = model.to(device)
    if dataset_name == "ImageNet":
        kwargs = {"split": "val"}
    elif "CIFAR" in dataset_name:
        kwargs = {"download": True, "train": False}
    elif "OxfordIIITPet" in dataset_name:
        kwargs = {"download": True}
    dataset: datasets.DatasetFolder = getattr(datasets, dataset_name)(
        data_dir, transform=preprocess, **kwargs)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    offload = rospy.get_param("/offload", True)
    offload_method = rospy.get_param("/offload_method", "all")

    from intraDP import intraDP, VideoFrameDataset
    offload_mode = rospy.get_param("/offload_mode", "DSCCS")
    ip = rospy.get_param("/server_ip", "127.0.0.1")
    port = rospy.get_param("/server_port", "12345")
    IDP = intraDP(offload=offload,
                                            parallel_approach=offload_method,
                                            ip=ip, port=port,
                                            constraint_latency=offload_mode=="SPSO-GA",
                                            log=rospy.loginfo)

    start = Event()
    correct_count = 0
    stime = time.time()
    from PIL import Image
    log_dir = os.environ["ROS_LOG_DIR"]
    idx = 0
    for i, (inp, target) in enumerate(dataloader):
        inp = inp.to(device)
        while True:
            if idx == 0:
                IDP.start_client([inp], {}, model, 0, "eth0", log_dir)
            pred = model(inp)
            # result = weights.meta['categories'][torch.argmax(pred[0])]
            # if result in dataset.classes[target]:
            #     correct_count += 1
            # if task == "classification" and i % 100 == 0:
            #     rospy.loginfo(f"Result: pred: {result}; gt: {dataset.classes[target]}.")
            #     rospy.loginfo(f"Accuracy@1 {correct_count / (i+1)*100:.4f}%")

            idx += 1
            rospy.loginfo(f"{idx} th input; finishes at {time.time()-stime:.4f} s")


# model candidates

# Classification
# name; param size; acc@1
# ResNet101; 44.5M; 81.886
# VGG19_BN; 143.7M; 74.218

# Semantic Segmentation
# name; param size; Mean IoU
# FCN; 54.3M; 63.7
# DeepLabV3; 61.0M; 67.4

# Object Detection
# name; param size; Box MAP
# FasterRCNN; 43.7M; 46.7
# RetinaNet; 38.2M; 41.5



