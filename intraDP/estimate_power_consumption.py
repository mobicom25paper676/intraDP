import subprocess
import time
from multiprocessing import Process, Value, Lock
import signal
import sys

def monitor_power_consumption(log_file, stop_flag, lock):
    """
    Monitor power consumption using tegrastats and log to a file.
    """
    # Command to run tegrastats (adjust the interval as needed)
    tegrastats_cmd = ["sudo", "tegrastats", "--interval", "3000"]  # 1000 ms = 1 second

    try:
        with open(log_file, "a") as f:
            # Start tegrastats process
            process = subprocess.Popen(tegrastats_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            while not stop_flag.value:
                # Read a line from tegrastats output
                output = process.stdout.readline()
                if output:
                    with lock:
                        f.write(output)  # Log the output to the file
                        f.flush()  # Ensure the data is written immediately
                else:
                    break

                # Sleep briefly to avoid high CPU usage
                time.sleep(0.1)

            # Terminate the tegrastats process
            print("Terminating tegrastats process...")
            process.terminate()
            process.wait()
            print("tegrastats process terminated.")
    except Exception as e:
        print(f"Error in monitor_power_consumption: {e}")
    finally:
        print("Power monitoring stopped.")


def init_power_monitor(log_file="power_consumption.log", interval=1000):
    stop_flag = Value('i', 0)  # Shared flag to stop the monitoring process
    lock = Lock()  # Lock to synchronize file access
    def signal_handler(sig, frame):
        """
        Handle Ctrl+C to stop the monitoring process gracefully.
        """
        print("Stopping power monitoring...")
        stop_flag.value = 1
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Start the power monitoring process
    monitor_process = Process(target=monitor_power_consumption, args=(log_file, stop_flag, lock), name="PowerMonitor")
    monitor_process.start()
    print(f"Monitoring power consumption. Logging to {log_file}...")

    # Register signal handler for Ctrl+C
    
    def monitor_stop():
        print("Stopping power monitoring...")
        stop_flag.value = 1
        monitor_process.join()
    return monitor_stop

def main():
    monitor_stop = init_power_monitor()
    time.sleep(10)
    monitor_stop()

if __name__ == "__main__":
    main()