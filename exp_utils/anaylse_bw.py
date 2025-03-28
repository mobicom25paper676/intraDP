import numpy as np

freq = 2
records = []
with open("/home/user/project/ParallelCollaborativeInference/exp_utils/indoors.txt", "r") as f:
    lines = f.readlines()
    for l in lines:
        records.append(float(l.split(" ")[-1]))
records = (np.array(records) / 8)
print(f"Mean {records.mean()}")
unique, count = np.unique(records.astype(int), return_counts=True)
total = count.sum()
d = dict(zip(unique, count))
# print(f"Count {d}")
print(f"Percentage {dict(zip(unique, np.round(count/total*100, decimals=2)))}")