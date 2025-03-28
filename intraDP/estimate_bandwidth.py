import subprocess
import time
from multiprocessing import Process, Value, Lock, Manager

def get_first_ethernet_interface():
    import psutil
    """Find the first Ethernet interface starting with 'eth'"""
    for interface in psutil.net_if_addrs():
        if interface.startswith("eth") or interface.startswith("eno"):
            return interface
    return None

def get_tc_parameters(interface):
    """
    Query the tc settings to get the configured bandwidth and network parameters.
    """
    params = {
        "bandwidth_bps": 0,
        "loss_rate": 0.0,
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "duplicate_rate": 0.0,
        "corrupt_rate": 0.0,
        "reorder_rate": 0.0,
    }

    try:
        # Get both qdisc and class information
        qdisc_output = subprocess.run(["tc", "qdisc", "show", "dev", interface], 
                                     capture_output=True, text=True).stdout
        class_output = subprocess.run(["tc", "class", "show", "dev", interface],
                                     capture_output=True, text=True).stdout

        # Parse qdisc output for netem parameters
        for line in qdisc_output.splitlines():
            if "netem" in line:
                if "loss" in line:
                    loss_str = line.split("loss ")[1].split("%")[0]
                    params["loss_rate"] = float(loss_str) / 100
                if "delay" in line:
                    delay_str = line.split("delay ")[1].split()[0]
                    params["delay_ms"] = float(delay_str.replace("ms", ""))
                    if " " in line.split("delay ")[1]:
                        params["jitter_ms"] = float(line.split("delay ")[1].split()[1].replace("ms", ""))
                if "rate" in line:
                    rate_str = line.split("rate ")[1].split()[0]
                    if "Gbit" in rate_str:
                        params["bandwidth_bps"] = float(rate_str.replace("Gbit", "")) * 1e9
                    elif "Mbit" in rate_str:
                        params["bandwidth_bps"] = float(rate_str.replace("Mbit", "")) * 1e6
                    elif "Kbit" in rate_str:
                        params["bandwidth_bps"] = float(rate_str.replace("Kbit", "")) * 1e3

        # Parse class output for bandwidth
        for line in class_output.splitlines():
            if "htb" in line and "rate" in line:
                rate_str = line.split("rate ")[1].split()[0]
                if "Gbit" in rate_str:
                    params["bandwidth_bps"] = float(rate_str.replace("Gbit", "")) * 1e9
                elif "Mbit" in rate_str:
                    params["bandwidth_bps"] = float(rate_str.replace("Mbit", "")) * 1e6
                elif "Kbit" in rate_str:
                    params["bandwidth_bps"] = float(rate_str.replace("Kbit", "")) * 1e3
                break  # Assume first class contains main rate

        return params
    except Exception as e:
        print(f"Error querying tc: {e}")
        return None


def bandwidth_monitor(interface, shared_params, lock, interval):
    """
    Periodically update the shared parameters.
    """
    while True:
        # Get the current tc parameters
        params = get_tc_parameters(interface)
        if params:
            with lock:
                for key, value in params.items():
                    setattr(shared_params, key, value)
        else:
            with lock:
                shared_params.bandwidth_bps = 0  # Indicate an error or no bandwidth

        # Wait for the next interval
        time.sleep(interval)

def estimate_transmission_time(N_bytes, params, print_debug=False):
    """
    Estimate the transmission time for N bytes given the tc parameters.
    """
    if params.bandwidth_bps <= 0:
        return None

    # Adjust for packet loss, duplication, corruption, and reordering
    effective_loss = params.loss_rate + params.duplicate_rate + params.corrupt_rate + params.reorder_rate
    effective_bandwidth = params.bandwidth_bps * (1 - effective_loss)
    # Calculate transmission time
    transmission_time = (N_bytes * 8) / effective_bandwidth
    # Add SPSO-GAed delay and jitter
    transmission_time += (params.delay_ms + params.jitter_ms) / 1000

    if print_debug:
        if params.bandwidth_bps > 0:
            print(f"Available bandwidth: {params.bandwidth_bps / 1e6:.2f} Mbps")
            print(f"Packet loss rate: {params.loss_rate * 100:.2f}%")
            print(f"Delay: {params.delay_ms:.2f} ms, Jitter: {params.jitter_ms:.2f} ms")
            print(f"Duplicate rate: {params.duplicate_rate * 100:.2f}%")
            print(f"Corrupt rate: {params.corrupt_rate * 100:.2f}%")
            print(f"Reorder rate: {params.reorder_rate * 100:.2f}%")
        print(f"Effective bandwidth: {effective_bandwidth / 1e6:.2f} Mbps")
        print(f"Estimated transmission time: {transmission_time:.2f} seconds")

    return transmission_time

def init_bandwidth_monitor(interval=3):
    interface = get_first_ethernet_interface()
    assert interface is not None, "No Ethernet interface found!"
    manager = Manager()
    shared_params = manager.Namespace()
    shared_params.bandwidth_bps = 0
    shared_params.loss_rate = 0.0
    shared_params.delay_ms = 0.0
    shared_params.jitter_ms = 0.0
    shared_params.duplicate_rate = 0.0
    shared_params.corrupt_rate = 0.0
    shared_params.reorder_rate = 0.0
    lock = Lock()

    # Start the bandwidth monitoring process
    monitor_process = Process(target=bandwidth_monitor, args=(interface, shared_params, lock, interval), name="BandwidthMonitor")
    monitor_process.start()
    
    def estimate_bandwidth():
        params = shared_params
        bandwidth_bps = params.bandwidth_bps
        if bandwidth_bps == 0:
            print("Unable to estimate transmission time (invalid parameters). Something is wrong with tc. Assuming 50 Mbps.")
            return 50 / 8
        
        return bandwidth_bps / 1e6 / 8  # View as bandwidth

    def monitor_stop():
        monitor_process.terminate()
        monitor_process.join()
    return estimate_bandwidth, monitor_stop

if __name__ == "__main__":
    estimate_bandwidth, stop_monitor = init_bandwidth_monitor(0.3)

    try:
        while True:
            estimate_bandwidth()
            print("Estimated bandwidth: ", estimate_bandwidth())
            # Simulate main thread doing other work
            time.sleep(1)  # Adjust as needed

    except KeyboardInterrupt:
        print("Stopping bandwidth monitor...")
        stop_monitor()
