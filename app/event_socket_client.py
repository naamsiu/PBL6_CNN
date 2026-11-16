import socket
import json


class AiEventClient:

    def __init__(self, path="/tmp/pbl6_ai.sock"):
        self.path = path

    def send(self, state):
        data = {"state": state}
        message = json.dumps(data) + "\n"

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.2)
                client.connect(self.path)
                client.sendall(message.encode("utf-8"))

            return True

        except OSError as e:
            print(f"[AI IPC] Error: {e}")
            return False