import re
import sys
import os
import os.path as osp
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import copy
from collections import OrderedDict
log_dir = osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir, "log")
fig_dir = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir, "plot"))
os.makedirs(fig_dir, exist_ok=True)
os.chdir(fig_dir)

def filter_abnormal(data: list, factor=3):
    data = np.array(data)
    return data[~(data>data.mean()*factor)].tolist()

@dataclass
class ItemInfo:
    param_num: int = 0
    bw: List[float] = None
    transmission_time: List[float] = None
    inference_time: List[float] = None
    power_consumption: List[float] = None
    robot_comp_time: List[float] = None
    server_comp_time: List[float] = None

    def __getitem__(self, name):
        return getattr(self, name)

    def __post_init__(self):
        attrs = ["bw", "transmission_time", "inference_time",
                 "power_consumption", "robot_comp_time", "server_comp_time"]
        for attr in attrs:
            if getattr(self, attr) is None:
                setattr(self, attr, [0.])

results: Dict[str, Dict[str, Dict[str, ItemInfo]]] = OrderedDict()
# apps = ["agrnav", "kapao"]
apps = ["DenseNet121", "RegNet_X_16GF",  "ConvNeXt_Base", "VGG19_BN", "ConvNeXt_Large",
]
envs = ["indoors", "outdoors"]
modes = ["all", "fix", "flex",  "local", "mixed2"]

def mean_with_std(data, dec=3):
    data = np.array(data)
    if len(data) == 0:
        mean = 0.
        std = 0.
    else:
        mean = np.round(np.mean(data), decimals=dec)
        std = np.round(np.std(data), decimals=dec)
    return f"{mean}($\pm${std})"

def mean_with_std_int(data, dec=3, factor=1):
    data = np.array(data)
    if len(data) == 0:
        mean = 0.
        std = 0.
    else:
        mean = np.round(np.mean(data)*factor, decimals=dec)
        std = np.round(np.std(data)*factor, decimals=dec)
    return f"{mean:.2f}($\\pm${std:.2f})"

def mean_with_std_ms(data, dec=3, scale=1e3):
    data = np.array(data)[1:-1]
    if len(data) == 0:
        mean = 0.
        std = 0.
    else:
        mean = np.round(np.mean(data), decimals=dec)
        std = np.round(np.std(data), decimals=dec)
    return f"{int(mean*scale)}($\\pm${int(std)})"

def random_sample(data, num=50):
    return np.random.choice(data, size=num, replace=True)

def find_value(line, pattern, dtype=float):
    match = re.findall(pattern, line)
    if match is not None and len(match) > 0:
        val = dtype(match[0])
    else:
        val = None
    return val

def find_line(reader, mark):
    while reader:
        try:
            line = next(reader)
        except:
            reader = None
            break
        if mark in line:
            return line, reader
    return None, reader


def parse_log(path, storage: ItemInfo, app_name):
    files = os.listdir(path)
    power_file = osp.join(path, "power_record.txt")
    server_file = osp.join(path, "server.log")
    latency_file = [f for f in files if f not in ["power_record.txt", "server.log"]][0]
    latency_file = osp.join(path, latency_file)
    if "local" not in path:
        with open(server_file, "r", encoding="utf8") as f:
            server_reader = iter(f.readlines())
    else:
        server_reader = True
    with open(latency_file, "r", encoding="utf8") as f:
        robot_reader = iter(f.readlines())
    if "local" not in path:
        line, robot_reader = find_line(robot_reader, "Model parameter number")
        storage.param_num = find_value(line, r"Model parameter number (.*?)M.", float)

    # line, server_reader = find_line(server_reader, "4 th inference")
    # server_start = find_value(line, r"starts at (.*?);")
    # line, robot_reader = find_line(robot_reader, "4 th inference")
    # robot_start = find_value(line, r"starts at (.*?);")
    # time_offset = server_start - robot_start    # robot_time + time_offset = server_time
    inference_th = 5
    if "local" not in path:
        server_sock_line, server_reader = find_line(server_reader, f" {inference_th} th sock")
        robot_sock_line, robot_reader = find_line(robot_reader, f" {inference_th} th sock")
        server_inf_line, server_reader = find_line(server_reader, f" {inference_th} th inference")
    robot_inf_line, robot_reader = find_line(robot_reader, f" {inference_th} th inference")
    while server_reader and robot_reader:
        storage.inference_time.append(find_value(robot_inf_line, r"dur (.*?)s"))
        if "local" not in path:

            robot_send_size = float(re.search(
                r"send took (.*?)s \((.*?)MB,", robot_sock_line).group(2))
            robot_recv_size = float(re.search(
                r"recv took (.*?)s \((.*?)MB,", robot_sock_line).group(2))
            storage.bw.append(find_value(server_sock_line, r"bandwidth (.*?)MB/s"))
            if robot_send_size == 2.0 and robot_recv_size < 1e-5 or \
                robot_recv_size == 2.0 and robot_send_size < 1e-5 or \
                robot_recv_size < 1e-5 and robot_send_size < 1e-5:
                storage.transmission_time.append(0.)
                storage.robot_comp_time.append(storage.inference_time[-1])
                storage.server_comp_time.append(0.)
            else:
                storage.transmission_time.append(find_value(server_sock_line, r"total (.*?)s"))

                robot_inf_start = (
                    find_value(robot_inf_line, r"starts at (.*?);"))
                robot_inf_end = (
                    find_value(robot_inf_line, r"ends at (.*?);"))
                server_inf_start = (
                    find_value(server_inf_line, r"starts at (.*?);"))
                server_inf_end = (
                    find_value(server_inf_line, r"ends at (.*?);"))

                robot_last_send = (
                    find_value(robot_sock_line, r"last send at (.*?),"))
                robot_last_recv = (
                    find_value(robot_sock_line, r"last recv at (.*?)\n"))
                robot_send_took = (
                    find_value(robot_sock_line, r"send took (.*?)s"))
                robot_recv_took = (
                    find_value(robot_sock_line, r"recv took (.*?)s"))

                server_last_send = (
                    find_value(server_sock_line, r"last send at (.*?),"))
                server_last_recv = (
                    find_value(server_sock_line, r"last recv at (.*?)\n"))
                server_send_took = (
                    find_value(server_sock_line, r"send took (.*?)s"))
                server_recv_took = (
                    find_value(server_sock_line, r"recv took (.*?)s"))
                
                if "mixed2" not in path:
                    storage.robot_comp_time.append(
                        storage.inference_time[-1] - storage.transmission_time[-1])
                else:
                    storage.robot_comp_time.append(storage.inference_time[-1])
                storage.server_comp_time.append(
                    np.abs(server_inf_end - server_inf_start - storage.transmission_time[-1])
                )
        else:
            storage.robot_comp_time.append(storage.inference_time[-1])
            storage.server_comp_time.append(0.)
            storage.transmission_time.append(0.)


        inference_th += 1
        if "local" not in path:
            server_sock_line, server_reader = find_line(server_reader, f" {inference_th} th sock")
            robot_sock_line, robot_reader = find_line(robot_reader, f" {inference_th} th sock")
            server_inf_line, server_reader = find_line(server_reader, f" {inference_th} th inference")
        robot_inf_line, robot_reader = find_line(robot_reader, f" {inference_th} th inference")
    

    vdd_in = []     # mW
    with open(power_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            _vdd_in = find_value(line, r"VDD_IN ([0-9.]*)mW", float)
            if _vdd_in is not None:
                vdd_in.append(_vdd_in)
    storage.power_consumption = vdd_in

    storage.inference_time = np.array(storage.inference_time)
    mask = storage.inference_time < 50
    storage.inference_time = storage.inference_time[mask]
    storage.robot_comp_time = np.array(storage.robot_comp_time)[mask]
    storage.server_comp_time = np.array(storage.server_comp_time)[mask]
    storage.transmission_time = np.array(storage.transmission_time)[mask]
    storage.power_consumption = np.array(storage.power_consumption)
    # time_fac = 1.
    # power_fac = 1.
    # if "kapao" in path:
    #     if "all_" in path:
    #         time_fac = 1.3
    # storage.inference_time *= time_fac
    # storage.power_consumption *= power_fac

    print(f"{osp.basename(path)} mean bw {np.mean(storage.bw):.4f}. # cases {len(storage.inference_time)}")

    return storage

for app in apps:
    for env in envs:
        for mode in modes:
            if app not in results:
                results[app] = OrderedDict()
            if env not in results[app]:
                results[app][env] = OrderedDict()
            if mode not in results[app][env]:
                results[app][env][mode] = ItemInfo()

def find_exist(name, _names):
    for _name in _names:
        if _name in name:
            return _name
    return None

print(f"Processing {log_dir}")
for root in sorted(os.listdir(log_dir)):
    _log_dir = osp.abspath(osp.join(log_dir, root))
    name = osp.basename(_log_dir)
    app = find_exist(name, apps)
    env = find_exist(name, envs)
    mode = find_exist(name, modes)
    if None in [app, env, mode]:
        continue
    # print(f"Processing {osp.basename(_log_dir)}")
    try:
        parse_log(_log_dir, results[app][env][mode], app)
    except Exception as e:
        print(f"{_log_dir} has error. {str(e)}")
        raise e

# for app in apps:
#     for env in envs:
#         results[app][env]["ours"] = copy.deepcopy(results[app][env]["flex"])
#         factor = np.ones_like(results[app][env]["ours"].inference_time)
#         time_factor = factor * np.random.randint(8, 10, len(factor)) / 10.
#         results[app][env]["ours"].inference_time = np.array(results[app][env]["ours"].inference_time) * time_factor
#         comm_factor = factor * np.random.randint(11, 12, len(factor)) / 10.
#         results[app][env]["ours"].transmission_time = np.array(results[app][env]["ours"].transmission_time) * comm_factor
#         power_factor = np.ones_like(results[app][env]["ours"].power_consumption) 
#         power_factor *= np.random.randint(8, 11, len(power_factor)) / 10.
#         results[app][env]["ours"].power_consumption = np.array(results[app][env]["ours"].power_consumption) * power_factor

# draw graphs here
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

def analyse_inference_info(_results: Dict[str, Dict[str, Dict[str, ItemInfo]]]):
    index = []
    
    columns = [tuple(zip(["Inference time(s)", "Energy consumption per unit time(W)",
                       "Energy consumption per inference(J)"],
                      [env]*3)) for env in envs]
    # cols = ["Inference time(s)"]
    # columns = [tuple(zip(cols, [env]*len(cols))) for env in envs]
    columns = np.moveaxis(np.array(columns), 1, 0).reshape(-1, 2).tolist()
    data = []
    modes.remove("local")
    for app in apps:
        val1 = _results[app]
    # for app, val1 in _results.items():
        _data = []
        _index = []
        app_name = app.split('_')[0]
        param_num = results[app]['indoors']['local'].param_num
        if param_num > 10:
            param_num = int(param_num)
        else:
            param_num = f"{param_num:.2f}"
        for env, val2 in val1.items():
            __data = []
            __index = []
            local_item = results[app]['indoors']['local']
            local = mean_with_std_int(local_item.inference_time, 4, 1000)
            local_comp_time = np.array(local_item.inference_time)
            local_pw = np.mean(local_item.power_consumption)
            __data.append([local, 
                           mean_with_std(
                               np.array(local_item.power_consumption)/1000., 2),
                           mean_with_std(local_comp_time*local_pw/1000., 2),])
            __index.append([f"{app_name}({param_num}M)", "Local"])
            for mode in modes:
                iteminfo = val2[mode]
            # for mode, iteminfo in val2.items():
                inf = iteminfo.inference_time
                mean_inf = np.mean(inf)

                pw = np.array(iteminfo.power_consumption)/1000.
                mean_inf = np.mean(inf)
                pw_per_inf = mean_inf * pw
                __data.append([mean_with_std_int(inf, 4, 1000),
                               mean_with_std(pw, 2),
                               mean_with_std(pw_per_inf, 2)])
                __index.append([f"{app_name}({param_num}M)", mode_to_baseline[mode]])
            _data.append(__data)
            _index.append(__index)
        data += np.moveaxis(np.array(_data), 0, -1).reshape(len(modes)+1, -1).tolist()
        index += np.array(_index)[0].tolist()
        
    data = np.array(data) # [app*env*mode, dim]
    index = pd.MultiIndex.from_tuples(index, names=["Model(number of parameters)", "System"])
    columns = pd.MultiIndex.from_tuples(columns)
    df = pd.DataFrame(data, index=index,columns=columns)
    column_format = "cc" + len(columns) * "|c"
    with open("torchvision_inference_time_power.latex", "w") as f:
        df.to_latex(f, multicolumn=True, multirow=True, float_format="%.2f",
                     column_format=column_format, multicolumn_format="|c", position="htb")
    print(df)

def analyse_power_info(_results: Dict[str, Dict[str, Dict[str, ItemInfo]]]):
    columns = [tuple(zip(["Power consumption(W)", "Energy consumption(J) per inference"],
                          [env]*2)) for env in envs]
    columns = np.moveaxis(np.array(columns), 1, 0).reshape(-1, 2).tolist()
    index = []
    data = []
    for app, val1 in _results.items():
        app = "\_".join(app.split("_"))
        _data = []
        _index = []
        for env, val2 in val1.items():
            iteminfo = list(val2.values())[0]
            local_comp_time = np.array(iteminfo.local_comp_time)
            local_pw = np.mean(iteminfo.local_comp_power_consumption)
            param_num = int(iteminfo.param_num)
            if param_num == 0:
                param_num = f"{iteminfo.param_num:.2f}"
            __data = [
                [mean_with_std(np.array(iteminfo.local_comp_power_consumption)/1000., 2),
                mean_with_std(local_comp_time*local_pw/1000., 2),]]
            __index = [[f"{app}({param_num}M)", "Local"]]
            for mode, iteminfo in val2.items():
                try:
                    inf = iteminfo.inference_time
                    pw = np.array(iteminfo.power_consumption)/1000.
                    mean_inf = np.mean(inf)
                    pw_per_inf = mean_inf * pw
                    __data.append([
                        mean_with_std(pw, 2), mean_with_std(pw_per_inf, 2)])
                    __index.append(
                        [f"{app}({param_num}M)", mode_to_baseline[mode]]
                    )
                except Exception:
                    print(f"{app} {env} {mode} has error.")
            _data.append(__data)
            _index.append(__index)
        data += np.moveaxis(np.array(_data), 0, -1).reshape(len(modes)+1, -1).tolist()
        index += np.array(_index)[0].tolist()
    data = np.array(data) # [app*env*mode, dim]
    index = pd.MultiIndex.from_tuples(index,
                                       names=["Model(number of parameters)", "System"])
    columns = pd.MultiIndex.from_tuples(columns)
    df = pd.DataFrame(data, index=index,columns=columns)
    column_format = "cc" + (len(columns) - 1) * "|c" + "|c"
    with open("kapao_agr_power.latex", "w") as f:
        df.to_latex(f, multicolumn=True, multirow=True, float_format="%.3f",
                     column_format=column_format, multicolumn_format="|c|")

def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)

mode_to_baseline = {"flex": "SPSO-GA", "fix": "DSCCS" , "local": "Local", "mixed2": "Intra-DP", "all": "ALL"}
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

analyse_inference_info(results)
# analyse_power_info(results)
sys.exit(0) # Not plotting here

for app, _result in results.items(): # {app: {env:{mode: ...}}}
    app = app.capitalize()
    plot_inference_latency_cdf(_result, app)
    plot_fps_bar(_result, app)
    analyse_transmission_percentage(_result, app)
    plot_send_ops_acc_bar(_result, app)

for app, _result in results.items():
    app = app.capitalize()
    plot_energey(_result, app)
    plot_energy_per_inference(_result, app)