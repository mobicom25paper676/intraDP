import sys
import time
import os
import random
import logging
import numpy as np
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

record_path = sys.argv[1]
wnic_name = sys.argv[2] # Can only use tc at server
scale = float(sys.argv[3])
if len(sys.argv) > 4:
    dry_run_len = int(sys.argv[4])
else:
    dry_run_len = 0
seed = 2233
seg_len = 5

random.seed(seed)

# # find wireless NIC
# wnic_name = None
# with open('/proc/net/dev') as f:
#     f_iter = iter(f)
#     next(f_iter)
#     next(f_iter)
#     for line in f_iter:
#         nic_name = line.split()[0][:-1]
#         if nic_name[:2] == 'wl':
#             assert wnic_name is None
#             wnic_name = nic_name
print(f'wnic_name {wnic_name}')
os.system(f' tc qdisc del dev {wnic_name} root')
os.system(f' tc qdisc del dev {wnic_name} ingress')

# read bandwidth record
bw_record = []  # format: [(time: float second, bw: float Mbps)]
with open(record_path, "r") as f:
    for line in f:
        try:
            bw_record.append(list(map(float, line.split())))
        except Exception as e:
            print(e)

def random_stop():
    num = random.randint(1, 20)
    if num > 10:
        stop_time = random.randint(10, 90)
        print(f'Stop for {stop_time} seconds')
        time.sleep(stop_time)

script_path = os.path.join(os.environ['work'], "exp_utils", "limit_bandwidth.sh")
def set_bandwidth(nic_name, bw):
    logging.info(f'{bw/8:.4f}MB/s')
    cmd = f'bash {script_path} {wnic_name} {int(bw*1024)} {int(bw*1024)}'
    if dry_run_len == 0:
        os.system(cmd)

import atexit
def exit_handler():
    logging.info(f"Average bw {sum(bws)/len(bws)/8:.4f}MB/s")
    os.system(f' tc qdisc del dev {wnic_name} root')
    os.system(f' tc qdisc del dev {wnic_name} ingress')
    os.system(" ip link del dev ifb1")
    os.system(" ip link del dev ifb0")
    logging.info("Removed tc queue")
atexit.register(exit_handler)

# replay the bandwidth
# to approximate real-world randomness, we randomly select seg_len segment to replay
cmd = f'bash ./limit_bandwidth.sh {wnic_name}'
os.system(cmd)
total_len = len(bw_record)
count = 0
bws = []
seg_num = 50
seg_count = 0
seg_interval = total_len // seg_num
start = random.randint(0, seg_interval)
idx = start
while True:
    bws.append(bw_record[idx][1])
    set_bandwidth(wnic_name, bw_record[idx][1])
    time.sleep(0.5)
    idx = (idx + 1) % total_len
    count += 1
    if count > seg_len:
        count = 0
        idx += seg_interval
        seg_count += 1
        if seg_count >= seg_num:
            seg_count = 0
            start = random.randint(0, seg_interval)
            idx = start


