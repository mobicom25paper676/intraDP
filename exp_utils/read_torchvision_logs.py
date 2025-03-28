import re
import sys
import os
import os.path as osp
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from collections import OrderedDict
log_dir = osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir, "log")
fig_dir = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir, "plot"))
os.makedirs(fig_dir, exist_ok=True)
os.chdir(fig_dir)
@dataclass
class ItemInfo:
    ops_num: int = 0
    param_num: int = 0
    input_size: float = 0.
    output_size: float = 0.
    send_ops: List[List[int]] = None
    bw: List[float] = None
    FPS_time: List[float] = None
    transmission_time: List[float] = None
    inference_time: List[float] = None
    power_consumption: List[float] = None
    local_comp_time: List[float] = None
    local_FPS_time: List[float] = None
    local_comp_power_consumption: List[float] = None

    def __getitem__(self, name):
        return getattr(self, name)

results: Dict[str, Dict[str, Dict[str, ItemInfo]]] = OrderedDict()
# apps = ["agrnav", "kapao"]
apps = ["DenseNet121", "RegNet_X_16GF",  "ConvNeXt_Base", "VGG19_BN", "ConvNeXt_Large",
]
envs = ["indoors", "outdoors"]
modes = ["fix", "flex", "mixed2"]

def align(l: list, factor: float):
    return (np.array(l) * factor ).tolist()

def mean_with_std(data, dec=3):
    data = np.array(data)
    if len(data) == 0:
        mean = 0.
        std = 0.
    else:
        mean = np.round(np.mean(data), decimals=dec)
        std = np.round(np.std(data), decimals=dec)
    return f"{mean}($\\pm${std})"

def mean_with_std_int(data, dec=3, factor=1):
    data = np.array(data)
    if len(data) == 0:
        mean = 0.
        std = 0.
    else:
        mean = np.round(np.mean(data), decimals=dec)
        std = np.round(np.std(data), decimals=dec)
    return f"{mean*factor:.1f}($\\pm${std*factor:.1f})"

def random_sample(data, num=50):
    return np.random.choice(data, size=num, replace=True)

def find_value(line, pattern, dtype=float):
    match = re.findall(pattern, line)
    if match is not None and len(match) > 0:
        val = dtype(match[0])
    else:
        val = None
    return val

def parse_local_log(path):
    files = os.listdir(path)
    try:
        power_idx = files.index("power_record.txt")
        power_file = osp.join(path, files[power_idx])
        files.remove(files[power_idx])
    except ValueError:
        power_file = None
    if "server.txt" in files:
        files.remove("server.txt")
    latency_file = osp.join(path, files[0])

    inference_time = []     # s
    FPS_time = []           # s
    with open(latency_file, "r", encoding="utf8") as f:
        _last_inference_start_time = None
        for line in f.readlines():
            if "inference" in line:
                _inference_start_time = find_value(line, r"starts at (.*?);")
                if _last_inference_start_time:
                    _fps_time = _inference_start_time - _last_inference_start_time
                else:
                    _fps_time = 0.
                _last_inference_start_time = _inference_start_time
                _inference_time = find_value(line, r"dur (.*?)s")

                inference_time.append(_inference_time)
                FPS_time.append(_fps_time)

    vdd_in = []     # mW
    if power_file:
        with open(power_file, "r", encoding="utf8") as f:
            for line in f.readlines():
                _vdd_in = find_value(line, r"VDD_IN ([0-9.]*)mW", float)
                if _vdd_in is not None:
                    vdd_in.append(_vdd_in)
    else:
        vdd_in = [0.]
    if len(inference_time) < 100 or len(FPS_time) < 100:
        print(f"{path} has error")
        inference_time = [0., 0.]
        FPS_time = [0., 0.]
    return inference_time, FPS_time, vdd_in

def parse_log(path, storage: ItemInfo, app_name):
    files = os.listdir(path)
    try:
        power_idx = files.index("power_record.txt")
        power_file = osp.join(path, files[power_idx])
        files.remove(files[power_idx])
    except ValueError:
        print(f"{path.split('/')[-1]} has no power record" )
        power_file = None
    if "server.txt" in files:
        files.remove("server.txt")
    latency_file = osp.join(path, files[0])
    
    send_ops = []
    transmission_time = []  # s
    inference_time = []     # s
    FPS_time = []           # s
    bw = []                 # MB/s
    ops_num = 0
    model_param_num = 0
    input_size = 0.         # MB
    output_size = 0.        # MB
    with open(latency_file, "r", encoding="utf8") as f:
        _send_ops = []
        _transmission_time = 0.
        _last_inference_start_time = None
        for i, line in enumerate(f.readlines()):
            if _ops_num := find_value(line, r"; total (.*?) ops", int):
                ops_num = _ops_num
            if _param_num := find_value(line, r"Model parameter number (.*?)M."):
                model_param_num = _param_num
            if _input_size := find_value(line, r"Input size (.*?)MB"):
                input_size = _input_size
            if _output_size := find_value(line, r"Output size (.*?)MB"):
                output_size = _output_size
            if (_send_op := find_value(line, r"op (.*?) [a-z_A-Z0-9]* send", int)) is not None:
                _send_ops.append(_send_op)
            if "inference est bw" in line:
                _inference_start_time = find_value(line, r"starts at (.*?);")
                if _last_inference_start_time:
                    _fps_time = _inference_start_time - _last_inference_start_time
                else:
                    _fps_time = 0.
                _last_inference_start_time = _inference_start_time
                _inference_time = find_value(line, r"dur (.*?)s")

                inference_time.append(_inference_time)
                FPS_time.append(_fps_time)
                send_ops.append(_send_ops)
                _send_ops = []
            if "send took" in line and "Init" not in line:
                if len(send_ops) > 0 and len(send_ops[len(transmission_time)-1]) > 0:
                    transmission_time.append(find_value(line, r"total (.*?)s"))
                    bw.append(find_value(line, r"bandwidth (.*?)MB/s"))
                else:
                    transmission_time.append(0.)
                    bw.append(0.)

    vdd_in = []     # mW
    if power_file:
        with open(power_file, "r", encoding="utf8") as f:
            for line in f.readlines():
                _vdd_in = find_value(line, r"VDD_IN ([0-9.]*)mW", float)
                if _vdd_in is not None:
                    vdd_in.append(_vdd_in)
    else:
        vdd_in = [0.]

    storage.ops_num = ops_num
    storage.param_num = model_param_num
    storage.input_size = input_size
    storage.output_size = output_size

    storage.send_ops = send_ops
    storage.bw = bw
    storage.FPS_time = FPS_time[3:]
    storage.transmission_time = transmission_time[3:]
    storage.inference_time = inference_time[3:]
    storage.power_consumption = vdd_in[3:]
    print(f"{osp.basename(path)} mean bw {np.mean(bw):.4f}.")

    storage.local_comp_time, storage.local_comp_power_consumption = [0.], [0.]
    for _path in os.listdir(osp.dirname(path)): # Read info of local computation
        if app_name in _path and "_local" in _path:
            local_comp_path = osp.join(osp.dirname(path), _path)
            storage.local_comp_time, storage.local_FPS_time, storage.local_comp_power_consumption =\
                parse_local_log(local_comp_path)
            # print(f"Processing local computation logs {local_comp_path}")
            
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

for root in sorted(os.listdir(log_dir)):
    _log_dir = osp.abspath(osp.join(log_dir, root))
    app = find_exist(_log_dir, apps)
    env = find_exist(_log_dir, envs)
    mode = find_exist(_log_dir, modes)
    if None in [app, env, mode] or "_local" in _log_dir:
        continue
    # print(f"Processing {_log_dir}")
    parse_log(_log_dir, results[app][env][mode], app)


# draw graphs here
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

def analyse_inference_info(_results: Dict[str, Dict[str, Dict[str, ItemInfo]]]):
    index = []
    columns = [tuple(zip(["Transmission time/s", "Inference time(s)", "Percentage(\%)"], [env]*3)) for env in envs]
    columns = np.moveaxis(np.array(columns), 1, 0).reshape(-1, 2).tolist()
    data = []
    for app, val1 in _results.items():
        app = "\_".join(app.split("_"))
        _data = []
        _index = []
        for env, val2 in val1.items():
            __data = []
            __index = []
            for mode, iteminfo in val2.items():
                param_num = int(iteminfo.param_num)
                if param_num == 0:
                    param_num = f"{iteminfo.param_num:.2f}"
                try:
                    inf = iteminfo.inference_time
                    trans = iteminfo.transmission_time
                    mean_trans = np.mean(trans)
                    mean_inf = np.mean(inf)
                    __data.append([mean_with_std_int(trans, 4, 1000), mean_with_std_int(inf, 4, 1000),
                                f"{mean_trans/mean_inf*100:.2f}"])
                    __index.append([f"{app}({param_num}M)",
                            mean_with_std_int(iteminfo.local_FPS_time, 4, 1000), mode_to_baseline[mode]])
                except Exception as e:
                    __data.append([0., 0., 0.])
                    __index.append([f"{app}({param_num}M)", 0., mode_to_baseline[mode]])
                    print(f"{app} {env} {mode} has error. {str(e)}")
            _data.append(__data)
            _index.append(__index)
        data += np.moveaxis(np.array(_data), 0, -1).reshape(len(modes), -1).tolist()
        index += np.array(_index)[0].tolist()
        
    data = np.array(data) # [app*env*mode, dim]
    index = pd.MultiIndex.from_tuples(index, names=["Model(number of parameters)", "Local computation time/s", "System"])
    columns = pd.MultiIndex.from_tuples(columns)
    df = pd.DataFrame(data, index=index,columns=columns)
    column_format = "ccc" + (len(columns) - 1) * "|c" + "|c"
    with open("torchvision_transmission_percentage.latex", "w") as f:
        df.to_latex(f, multicolumn=True, multirow=True, float_format="%.2f",
                     column_format=column_format, multicolumn_format="|c|")

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
            param_num = int(iteminfo.param_num)
            if param_num == 0:
                param_num = f"{iteminfo.param_num:.2f}"
            local_comp_time = np.array(iteminfo.local_comp_time)
            try:
                local_pw = np.mean(iteminfo.local_comp_power_consumption)
                __data = [
                    [mean_with_std(np.array(iteminfo.local_comp_power_consumption)/1000., 2),
                    mean_with_std(local_comp_time*local_pw/1000., 2),]]
                __index = [[f"{app}({param_num}M)", "Local"]]
            except:
                __data = [[0., 0.]]
                __index = [[f"{app}({param_num}M)", "Local"]]
            for mode, iteminfo in val2.items():
                try:
                    inf = iteminfo.inference_time
                    pw = np.array(iteminfo.power_consumption)/1000.
                    mean_inf = np.mean(inf)
                    pw_per_inf = mean_inf * pw
                    __data.append([
                        mean_with_std(pw, 2), mean_with_std(pw_per_inf, 2)])
                except Exception as e:
                    __data.append([0., 0.])
                    print(f"{app} {env} {mode} has error.{str(e)}")
                __index.append(
                    [f"{app}({param_num}M)", mode_to_baseline[mode]])
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
    with open("torchvision_power.latex", "w") as f:
        df.to_latex(f, multicolumn=True, multirow=True, float_format="%.3f",
                     column_format=column_format, multicolumn_format="|c|")

def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)

mode_to_baseline = {"fix": "SPSO-GA", "flex": "DSCCS" , "Local": "Local", "mixed2": "Intra-DP"}
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
analyse_power_info(results)
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