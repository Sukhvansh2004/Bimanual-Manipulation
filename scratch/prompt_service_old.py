#!/usr/bin/env python3
"""
Prompt Service for managing FastSAM text prompts.
Provides TCP socket interface for dynamic prompt management.
"""

import socket
import threading
import time
from typing import Optional


class PromptService:
    """
    TCP socket service for managing FastSAM prompts.
    Allows dynamic changing of text prompts for object detection.
    """

    def __init__(self, host: str = 'localhost', port: int = 5557):
        self.host = host
        self.port = port
        self.current_prompt = "object"
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the prompt service."""
        if self.running:
            print("Prompt service already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"Prompt service started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the prompt service."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.thread:
            self.thread.join(timeout=1.0)
        print("Prompt service stopped")

    def get_current_prompt(self) -> str:
        """Get the current prompt."""
        return self.current_prompt

    def set_prompt(self, prompt: str) -> str:
        """Set a new prompt."""
        self.current_prompt = prompt
        return f"Prompt set to: {prompt}"

    def _run_server(self) -> None:
        """Run the TCP server loop."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Allow checking self.running

            print(f"Prompt service listening on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"Prompt service: Connection from {address}")
                    self._handle_client(client_socket)
                except socket.timeout:
                    continue
                except OSError:
                    break  # Socket was closed

        except Exception as e:
            print(f"Prompt service error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def _handle_client(self, client_socket: socket.socket) -> None:
        """Handle a client connection."""
        try:
            data = client_socket.recv(1024).decode('utf-8').strip()

            if not data:
                response = "ERROR: Empty command"
            elif data.upper() == "GET":
                response = f"CURRENT_PROMPT: {self.current_prompt}"
            elif data.upper().startswith("SET:"):
                prompt = data[4:].strip()
                if prompt:
                    response = self.set_prompt(prompt)
                else:
                    response = "ERROR: Empty prompt"
            else:
                response = "ERROR: Invalid command. Use 'GET' or 'SET: <prompt>'"

            client_socket.send(response.encode('utf-8'))

        except Exception as e:
            print(f"Error handling client: {e}")
            try:
                client_socket.send(f"ERROR: {e}".encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()


def main():
    """Run the prompt service standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Prompt Service for FastSAM")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5557, help="Port to bind to")

    args = parser.parse_args()

    service = PromptService(host=args.host, port=args.port)
    service.start()

    try:
        print("Prompt service running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping prompt service...")
        service.stop()


if __name__ == "__main__":
    main()