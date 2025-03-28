import subprocess
import sys
import os
import signal
import atexit
import psutil

def get_first_ethernet_interface():
    """Find the first Ethernet interface starting with 'eth'"""
    for interface in psutil.net_if_addrs():
        if interface.startswith("eth") or interface.startswith("eno"):
            return interface
    return None

def start_replay_bandwidth(file_path, log=print):
    """Start replay_bandwidth.py in the background and return a stop handler"""
    
    eth_interface = get_first_ethernet_interface()
    if not eth_interface:
        log("No Ethernet interface found!", file=sys.stderr)
        sys.exit(1)

    assert file_path is not None and os.path.exists(file_path), f"File {file_path} does not exist!"
    command = ["bash", os.path.join(os.environ['work'], "exp_utils", "apply_tc.sh"), f"{file_path}"]
    log(f"Starting background process: {' '.join(command)}")

    # Start the process in a new process group to ensure it stops with the main process
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               preexec_fn=os.setpgrp)  # Create new process group on Linux/Unix

    def stop_process():
        """Stops the background process"""
        if process:
            log("Stopping background process...")
            process.terminate()  # Sends SIGTERM
            try:
                process.wait(timeout=5)  # Wait for clean exit
            except subprocess.TimeoutExpired:
                log(f"Process {process.pid} did not terminate, killing...")
                process.kill()  # Force kill

    return stop_process  # Return the stop handler so it can be explicitly called

# Example Usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <env>")
        sys.exit(1)

    env = sys.argv[1]
    stop_handler = start_replay_bandwidth(env)

    # Setup signal handling for graceful exit
    def handle_signal(signum, frame):
        print(f"Received signal {signum}, stopping process...")
        stop_handler()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while True:
            signal.pause()  # Wait for signals
    except KeyboardInterrupt:
        print("Shutting down...")
        stop_handler()