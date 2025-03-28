import sys
from intraDP import intraDP

if __name__ == '__main__':
    ip = "192.168.50.11"
    port = 12345
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    if len(sys.argv) > 3:
        bandwidth_file_path = str(sys.argv[3])
    else:
        bandwidth_file_path = None
    IDP = intraDP(parallel_approach="select",
                                         ip=ip, port=port)
    IDP.start_server(bandwidth_file_path=bandwidth_file_path)
