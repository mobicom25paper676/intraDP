import re
import os
import os.path as osp
from collections import OrderedDict
log_dir = osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir, "log")
fig_dir = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir, "plot"))
os.makedirs(fig_dir, exist_ok=True)
os.chdir(fig_dir)

results = OrderedDict()
apps = ["agrnav", "kapao"]
envs = ["indoors", "outdoors"]
modes = ["fix", "flex"]

def random_sample(data, num=50):
    return np.random.choice(data, size=num, replace=True)

def find_value(line, pattern, dtype=float):
    match = re.findall(pattern, line)
    if match is not None and len(match) > 0:
        val = dtype(match[0])
    else:
        val = None
    return val

def parse_log(path, storage: dict):
    files = os.listdir(path)
    power_idx = files.index("power_record.txt")
    power_file = osp.join(path, files[power_idx])
    files.remove(files[power_idx])
    latency_file = osp.join(path, files[0])
    
    send_ops = []
    send_data = []          # MB
    receive_data = []       # MB
    transmission_time = []  # s
    inference_time = []     # s
    FPS_time = []           # s
    ops_num = 0
    with open(latency_file, "r", encoding="utf8") as f:
        _send_data = 0.
        _receive_data = 0.
        _send_ops = []
        for line in f.readlines():
            all_ops = re.findall(r"Operation records: \[(.*?)\]", line)
            if all_ops is not None and len(all_ops) > 0:
                ops_num = len(all_ops[0].split(", "))
            op_match = re.findall(r"Sending reuslts of op: \[(.*?)\]", line)
            if op_match is not None and len(op_match) > 0:
                op_match = op_match[0]
                ops = []
                # ops = [int(op) for op in op_match.split(" ")]
                for op in op_match.split(" "):
                    if len(op) > 0:
                        ops.append(int(op))
                send_ops.append(ops)
                _send_ops = ops
                if _send_data > 0:
                    send_data.append(_send_data)
                    receive_data.append(_receive_data)
                    _send_data = 0.
                    _receive_data = 0.
            __send = find_value(line, r"offload ([0-9.]*)MB at", float)
            if __send is not None:
                _send_data += __send
            __recv = find_value(line, r"recved ([0-9.]*)MB at", float)
            if __recv is not None:
                _receive_data += __recv
            _fps_time = find_value(line, r"Process last frame took: ([0-9.]*)s", float)
            _inference_time = find_value(line, r"total inference time: ([0-9.]*)s", float)
            _transmission_time = find_value(line, r"transmission time: ([0-9.]*)s", float)
            if _fps_time is not None:
                if len(_send_ops) > 0:
                    transmission_time.append(_transmission_time)
                else:
                    transmission_time.append(0.)
                FPS_time.append(_fps_time)
                inference_time.append(_inference_time)
                _send_ops = []
    size_diff = len(transmission_time) - len(send_ops)
    assert ops_num > 0
    if size_diff > 0:
        for _ in range(size_diff):
            send_ops.append([ops_num])
    storage["send_ops"] = send_ops
    storage["send_data"] = send_data
    storage["recv_data"] = receive_data
    storage["transmission_time"] = transmission_time
    storage["inference_time"] = inference_time
    storage["FPS_time"] = FPS_time
    storage["ops_num"] = ops_num

    vdd_in = []     # mW
    with open(power_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            _vdd_in = find_value(line, r"VDD_IN ([0-9.]*)mW", float)
            if _vdd_in is not None:
                vdd_in.append(_vdd_in)
    storage["power_consumption"] = vdd_in
    return storage

for root in sorted(os.listdir(log_dir)):
    if "2" not in root:
        continue
    _log_dir = osp.abspath(osp.join(log_dir, root))
    print(f"Processing {_log_dir}")
    for app in apps:
        if app in root:
            if app not in results:
                results[app] = OrderedDict()
            break
    for env in envs:
        if env in root:
            if env not in results[app]:
                results[app][env] = OrderedDict()
            break
    for mode in modes:
        if mode in root:
            if mode not in results[app][env]:
                results[app][env][mode] = OrderedDict()
            break
    _result = results[app][env][mode]
    results[app][env][mode] = parse_log(_log_dir, _result)


# TODO draw graphs here
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)

mode_to_baseline = {"fix": "DSCCS", "flex": "SPSO-GA"}
env_to_env = {""}
sns.set_context("poster")
sns.set_style(style="whitegrid")
sns.set_color_codes(palette="deep")
def plot_inference_latency_cdf(storage: dict, app):
    # storage: {env:{mode: ...}}
    plt.figure(figsize=(10, 6))
    data = {
            "offloading system & environment": [],
            "inference latency(second)": []}
    for env, _d in storage.items():
        for mode, d in _d.items():
            _data = d["inference_time"]
            data["offloading system & environment"] += [f"{mode_to_baseline[mode]} & {env}"] * len(_data)
            data["inference latency(second)"] += _data
    df = pd.DataFrame(data=data)
    p = sns.ecdfplot(df, x="inference latency(second)", stat="percent", hue="offloading system & environment")
    p.set_ylabel("Percent(%)")
    p.set_title(f"{app}: Inference Latency CDF")
    plt.savefig(f"{app}_inference_cdf.jpg", bbox_inches="tight")

def plot_fps_bar(storage: dict, app):
    plt.figure(figsize=(10, 6))
    data = {"environment": [],
            "offloading system": [],
            "Application FPS(Hz)": []}
    for env, _d in storage.items():
        for mode, d in _d.items():
            _data = np.array(d["FPS_time"])
            _data = _data[np.nonzero(_data)]
            data["offloading system"] += [mode_to_baseline[mode]] * len(_data)
            data["environment"] += [env] * len(_data)
            data["Application FPS(Hz)"] += (1. / _data).tolist()
    df = pd.DataFrame(data=data)
    p = sns.boxplot(df, x="offloading system",
                    y="Application FPS(Hz)", hue="environment",
                    whis=2., showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"15"})
    p.set_title(f"{app}: Frame Per Second")

    p.legend(loc="lower center", fontsize='16')
    plt.savefig(f"{app}_fps.jpg", bbox_inches="tight")

def analyse_transmission_percentage(storage: dict, app):
    plt.figure(figsize=(10, 6))
    index = []
    data = []
    for env, _d in storage.items():
        for mode, d in _d.items():
            mean_transmission_time = np.array(d["transmission_time"]).sum() / len(d["inference_time"])
            mean_inference_time = np.array(d["inference_time"]).mean()
            mean_transmitted_data = np.array(d["send_data"]).mean()
            mean_received_data = np.array(d["recv_data"]).mean()
            index.append((env, mode_to_baseline[mode]))
            data.append([mean_transmitted_data, mean_received_data, mean_transmission_time, mean_inference_time, mean_transmission_time/mean_inference_time*100])
            print(f"{app} {mode_to_baseline[mode]} {env}: mean inference time {mean_inference_time:.3f} ; mean transmission time {mean_transmission_time:.3f} ; percentage {mean_transmission_time / mean_inference_time * 100:.2f}")
    index = pd.MultiIndex.from_tuples(index, names=["environment", "offloading system"])
    df = pd.DataFrame(data, index=index,
                     columns=["transmitted data/MB", "received data/MB", "transmission time/s", "Inference time(s)", "percentage(%)"])
    with open(f"{app}_transmission_percentage.latex", "w") as f:
        df.T.to_latex(f, multicolumn=True, multirow=True, float_format="%.3f", column_format='c|c|c|c|c')

def plot_send_ops_acc_bar(storage: dict, app):
    fig = plt.figure(figsize=(10, 6))
    data = {
            "offloading system & environment": [],
            "Percent(%)": [],
            "offloading category": []}
    for env, _d in storage.items():
        for mode, d in _d.items():
            _data =  np.hstack(d["send_ops"])
            start_num = np.count_nonzero(_data == -1) / len(_data) * 100
            end_num = np.count_nonzero(_data == d["ops_num"]) / len(_data) * 100
            mid_num = np.count_nonzero((_data != -1) & (_data != d["ops_num"])) / len(_data) * 100
            
            data["offloading system & environment"] += [f"{mode_to_baseline[mode]} & {env}"] * 3
            # data["environment"] += [env] * 3
            data["Percent(%)"] += [end_num, mid_num, start_num]
            data["offloading category"] += ["local computation", "middle", "start"]
    df = pd.DataFrame(data=data)
    p = sns.histplot(df, x="offloading system & environment", weights="Percent(%)", hue="offloading category", multiple='stack', shrink=0.6)
    
    move_legend(p, (1,0), fontsize="16", title_fontsize= '18' )
    p.set_ylabel("Percent(%)")
    p.set_title(f"{app}: Offloading Category and Percentage")
    plt.xticks(rotation=30)
    plt.savefig(f"{app}_offloading_op_acc_bar.jpg", bbox_inches="tight")

for app, _result in results.items(): # {app: {env:{mode: ...}}}
    app = app.capitalize()
    plot_inference_latency_cdf(_result, app)
    plot_fps_bar(_result, app)
    analyse_transmission_percentage(_result, app)
    plot_send_ops_acc_bar(_result, app)

def plot_energey(storage: dict, app):
    plt.figure(figsize=(10, 6))
    data = {
        "environment": [],
        "offloading system": [],
        "power consumption(W)": []
    }
    for env, _d in storage.items():
        for mode, d in _d.items():
            data["power consumption(W)"].append(np.array(d["power_consumption"]).mean()/1000.)
            data["environment"].append(env)
            data["offloading system"].append(mode_to_baseline[mode])
    df = pd.DataFrame(data)
    p = sns.barplot(df, x="offloading system", y="power consumption(W)", hue="environment")
    for bar in p.patches:
        bar.set_hatch("//")
        bar.set_edgecolor('k')
    p.set_title(f"{app}: Runtime Energy Consumption")
    p.legend(loc="lower right")
    plt.savefig(f"{app}_power_consumption.jpg", bbox_inches="tight")

def plot_energy_per_inference(storage: dict, app):
    plt.figure(figsize=(10, 6))
    data = {
        "environment": [],
        "offloading system": [],
        "power consumption per frame(J)": []
    }
    for env, _d in storage.items():
        for mode, d in _d.items():
            energy = np.array(d["power_consumption"]).mean() * np.array(d["FPS_time"]).mean() / 1000.
            data["power consumption per frame(J)"].append(energy)
            data["environment"].append(env)
            data["offloading system"].append(mode_to_baseline[mode])
    df = pd.DataFrame(data)
    p = sns.barplot(df, x="offloading system",
                    y="power consumption per frame(J)", hue="environment")
    
    for bar in p.patches:
        bar.set_hatch("//")
        bar.set_edgecolor('k')
    p.set_title(f"{app}: Energy Consumption Per Frame")
    p.legend(loc="lower right")
    plt.savefig(f"{app}_power_consumption_per_frame.jpg", bbox_inches="tight")

for app, _result in results.items():
    app = app.capitalize()
    plot_energey(_result, app)
    plot_energy_per_inference(_result, app)