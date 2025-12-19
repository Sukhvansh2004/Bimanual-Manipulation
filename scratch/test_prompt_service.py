#!/usr/bin/env python3
"""
Helper script for testing the prompt service.
This demonstrates how to set and get prompts from the prompt service.
"""

import socket
import time
import sys


def send_command(host, port, command):
    """Send a command to the prompt service and return response."""
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        client.send(command.encode('utf-8'))
        response = client.recv(1024).decode('utf-8')
        client.close()
        return response.strip()
    except Exception as e:
        return f"Error: {e}"


def test_prompt_service(host='localhost', port=5557):
    """Test the prompt service functionality."""
    print(f"Testing prompt service at {host}:{port}")
    print("=" * 50)

    # Test GET command
    print("1. Getting current prompt...")
    response = send_command(host, port, "GET")
    print(f"   Response: {response}")

    # Test SET command
    print("\n2. Setting prompt to 'red cube'...")
    response = send_command(host, port, "SET: red cube")
    print(f"   Response: {response}")

    # Test GET again
    print("\n3. Getting current prompt again...")
    response = send_command(host, port, "GET")
    print(f"   Response: {response}")

    # Test another SET
    print("\n4. Setting prompt to 'blue bottle'...")
    response = send_command(host, port, "SET: blue bottle")
    print(f"   Response: {response}")

    # Final GET
    print("\n5. Final prompt check...")
    response = send_command(host, port, "GET")
    print(f"   Response: {response}")

    print("\n" + "=" * 50)
    print("Prompt service test complete!")


def interactive_prompt_setter(host='localhost', port=5557):
    """Interactive prompt setter for manual testing."""
    print(f"Interactive prompt setter for {host}:{port}")
    print("Type 'quit' to exit")
    print("-" * 30)

    while True:
        prompt = input("Enter new prompt: ").strip()
        if prompt.lower() == 'quit':
            break

        if prompt:
            command = f"SET: {prompt}"
            response = send_command(host, port, command)
            print(f"Response: {response}")

            # Also show current prompt
            current = send_command(host, port, "GET")
            print(f"Current prompt: {current}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_prompt_setter()
    else:
        test_prompt_service()