import socket
import threading
import multiprocessing
import queue
import time
import pickle
import netifaces
import io
import torch

def serialize(obj):
    """Serialize any torch tensor (CPU or GPU) to bytes."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)  # ✅ Save to memory buffer
    return buffer.getvalue()  # Return raw byte data

def deserialize(byte_data):
    """Deserialize bytes back into a torch tensor (CPU or GPU)."""
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer)  # ✅ Restore from memory buffer

def get_first_ethernet_ip():
    """Finds the first non-loopback Ethernet interface IP."""
    for iface in netifaces.interfaces():
        if iface.startswith(("eth", "en")):  # Match Ethernet or WiFi adapters
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:  # Check for IPv4 Address
                return addrs[netifaces.AF_INET][0]['addr']
    return None  # No suitable interface found

class BiDirectionalSocket:
    def __init__(self, address="127.0.0.1", port=5555, mode="server", log=print):
        """Runs both sending & receiving inside a new process using threads."""
        self.address = address
        self.port = port
        self.mode = mode.lower()
        self.log = log
        self.running = multiprocessing.Event()
        self.running.set()
        self.peername = None

        # Queues for sending & receiving (shared between main & worker process)
        self.send_queue = multiprocessing.Manager().Queue()
        self.recv_queue = multiprocessing.Manager().Queue()

        # Start a single process for TCP socket communication (runs threads internally)
        self.process = multiprocessing.Process(target=self.socket_process, daemon=True)
        self.process.start()

    def socket_process(self):
        """A new process that internally runs send & receive using threads."""
        if self.mode == "server":
            self.server_socket()
        else:
            self.client_socket()

    def server_socket(self):
        """Server listens for a connection, then runs send & receive threads."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.address, self.port))
        server_sock.listen(1)
        server_sock.settimeout(1.0)  # Prevents blocking forever in accept()

        self.log(f"[SERVER] Listening on {self.address}:{self.port}...")
        
        while self.running.is_set():
            try:
                server_sock.settimeout(1.0)  # Short timeout to allow checking self.running
                conn, addr = server_sock.accept()
                conn.settimeout(None)  # Timeout to avoid blocking indefinitely
                self.peername = addr
                self.log(f"[SERVER] Connected to {addr}")

                self.run_socket_threads(conn)
            except socket.timeout:
                continue

        server_sock.close()
        self.log(f"[SERVER] Shutdown complete.")


    def client_socket(self):
        """Client connects to the server and starts send & receive threads."""
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Automatically find Ethernet/WiFi interface IP
        if self.address.endswith("127.0.0.1"):
            bind_ip = None
        else:
            bind_ip = get_first_ethernet_ip()
        if not bind_ip:
            self.log("[CLIENT] No Ethernet interface found. Using default routing.")
        else:
            try:
                client_sock.bind((bind_ip, 0))  # Bind to Ethernet IP, random port
                self.log(f"[CLIENT] Bound to interface {bind_ip}")
            except OSError as e:
                self.log(f"[CLIENT] Failed to bind to {bind_ip}: {e}")
                return
        while self.running.is_set():
            try:
                client_sock.connect((self.address, self.port))
                self.peername = (self.address, self.port)
                break
            except ConnectionRefusedError:
                self.log("[CLIENT] Server not available, retrying...")
                time.sleep(1)

        self.log(f"[CLIENT] Connected to server {self.address}:{self.port}")

        self.run_socket_threads(client_sock)

    def run_socket_threads(self, sock):
        """Starts independent send and receive threads."""
        recv_t = threading.Thread(target=self.receive_thread, args=(sock,), daemon=True)
        send_t = threading.Thread(target=self.send_thread, args=(sock,), daemon=True)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        recv_t.start()
        send_t.start()

        recv_t.join()
        send_t.join()

        sock.close()
        self.log(f"[{self.mode.upper()}] Socket closed.")

    def receive_thread(self, sock):
        """Thread for receiving messages using a buffer to store extra data."""
        buffer = bytearray()

        while self.running.is_set():
            try:
                # Read as much data as possible to avoid multiple recv() calls
                data = sock.recv(16384)
                if not data:
                    self.log(f"[{self.mode.upper()}] Connection closed.")
                    self.running.clear()
                    break

                buffer.extend(data)  
                
                while len(buffer) >= 4:  # Process one message at a time
                    message_length = int.from_bytes(buffer[:4], byteorder="big")

                    if len(buffer) >= 4 + message_length:  # Ensure full message is received
                        message = buffer[4 : 4 + message_length]
                        self.recv_queue.put(message)  # Store message in queue
                        buffer = buffer[4 + message_length:]  # Remove processed message
                    else:
                        break  # Wait for more data if message isn't fully received

            except (ConnectionResetError, BrokenPipeError):
                self.log(f"[{self.mode.upper()}] Connection lost.")
                self.running.clear()
                break

    def send_thread(self, sock):
        """Thread for sending messages via the socket."""
        while self.running.is_set():
            try:
                message_to_send = self.send_queue.get(timeout=0.5)  # Non-blocking queue check

                length = len(message_to_send).to_bytes(4, byteorder="big")  # Send length first
                sock.sendall(length + message_to_send)

            except queue.Empty:
                pass  # No message to send
            except (ConnectionResetError, BrokenPipeError):
                self.log(f"[{self.mode.upper()}] Connection lost.")
                self.running.clear()
                break

    def stop(self):
        """Stops the process & releases resources."""
        self.running.clear()
        self.process.terminate()
        time.sleep(1.)
        self.process.kill()
        self.log(f"[{self.mode.upper()}] Stopped.")

    def send_message(self, message):
        """Send a message to the ZMQ process via queue."""
        msg = pickle.dumps(message)
        self.send_queue.put(msg)
        return len(msg)

    def send_raw_message(self, message):
        """Send a raw message to the ZMQ process via queue."""
        self.send_queue.put(message)    

    def receive_message(self):
        """Retrieve a received message from the queue (non-blocking)."""
        while True:
            try:
                msg = self.recv_queue.get(timeout=0.5)
                if msg:
                    return pickle.loads(msg)
                else:
                    raise EOFError
            except queue.Empty:
                if not self.running.is_set():
                    return None

    def receive_message_len(self):
        """Retrieve a received message from the queue (non-blocking)."""
        while True:
            try:
                msg = self.recv_queue.get(timeout=0.5)
                if msg:
                    return pickle.loads(msg), len(msg)
                else:
                    raise EOFError
            except queue.Empty:
                if not self.running.is_set():
                    return None, 0

    def receive_message_timeout(self, timeout=1.):
        """Retrieve a received message from the queue (non-blocking)."""
        while True:
            try:
                msg = self.recv_queue.get(timeout=timeout)
                if msg:
                    return pickle.loads(msg)
                else:
                    raise EOFError
            except queue.Empty:
                return ""

    def receive_raw_message(self):
        """Retrieve a received raw message from the queue (non-blocking)."""
        while True:
            try:
                msg = self.recv_queue.get(timeout=0.5)
                if msg:
                    return msg
                else:
                    raise EOFError
            except queue.Empty:
                if not self.running.is_set():
                    return None

if __name__ == "__main__":
    try:
        server = BiDirectionalSocket(mode="server")
        client = BiDirectionalSocket(mode="client")

        print("\n[INFO] Type messages and press Enter to send. Type 'exit' to quit.")

        while True:


            message = input("\n[YOU] Enter message: ").strip()
            if message.lower() == "exit":
                break

            # Send messages from main thread
            server.send_message(f"Server says: {message}")
            client.send_message(f"Client says: {message}")

            # Fetch messages received by server & client
            server_received = server.receive_message()
            client_received = client.receive_message()
            
            if server_received:
                print(f"\n[SERVER] Received: {server_received}")

            if client_received:
                print(f"\n[CLIENT] Received: {client_received}")

    except KeyboardInterrupt:
        print("\n[INFO] CTRL+C detected, shutting down gracefully.")
    finally:
        server.stop()
        client.stop()
