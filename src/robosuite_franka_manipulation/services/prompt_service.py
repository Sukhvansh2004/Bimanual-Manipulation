#!/usr/bin/env python3
"""
Microservice for setting text prompts for SAM object detection.
Runs a socket server that accepts text prompts and serves them to clients.
"""

import socket
import threading
import time
from typing import Optional

class PromptService:
    """
    Socket-based microservice for managing SAM text prompts.
    """
    def __init__(self, host: str = 'localhost', port: int = 5557):
        self.host = host
        self.port = port
        self.current_prompt: str = "object"  # Default prompt
        self.server_socket: Optional[socket.socket] = None
        self.running = False

    def start(self) -> None:
        """Start the prompt service server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        print(f"Prompt service started on {self.host}:{self.port}")
        print(f"Default prompt: '{self.current_prompt}'")

        # Start server thread
        server_thread = threading.Thread(target=self._server_loop)
        server_thread.daemon = True
        server_thread.start()

    def stop(self) -> None:
        """Stop the prompt service."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _server_loop(self) -> None:
        """Main server loop accepting connections."""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                client_thread = threading.Thread(target=self._handle_client, args=(client_socket, addr))
                client_thread.daemon = True
                client_thread.start()
            except OSError:
                break  # Socket closed
            except Exception as e:
                print(f"Server error: {e}")

    def _handle_client(self, client_socket: socket.socket, addr) -> None:
        """Handle individual client connections."""
        try:
            data = client_socket.recv(1024).decode('utf-8').strip()
            if data:
                if data.startswith("SET:"):
                    # Set new prompt
                    new_prompt = data[4:].strip()
                    self.current_prompt = new_prompt
                    print(f"Updated prompt to: '{self.current_prompt}'")
                    client_socket.send(b"OK")
                elif data == "GET":
                    # Get current prompt
                    client_socket.send(self.current_prompt.encode('utf-8'))
                else:
                    client_socket.send(b"ERROR: Invalid command")
            else:
                client_socket.send(self.current_prompt.encode('utf-8'))
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            client_socket.close()

    def get_current_prompt(self) -> str:
        """Get the current text prompt."""
        return self.current_prompt

def main():
    """Main function to run the prompt service."""
    import argparse

    parser = argparse.ArgumentParser(description="SAM Prompt Service")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5557, help="Port to bind to")

    args = parser.parse_args()

    service = PromptService(host=args.host, port=args.port)
    service.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping prompt service...")
        service.stop()

if __name__ == '__main__':
    main()