"""
Natural Language Command Interface for SemanticVLN-MCP
========================================================
Run this alongside Webots to send commands to the robot.
"""

import sys
import os

# Add project root
sys.path.insert(0, r"c:\Users\Lenovo\Documents\new_pro")

import asyncio
import json
from pathlib import Path

# Command file for IPC with Webots
COMMAND_FILE = Path(r"c:\Users\Lenovo\Documents\new_pro\robot_command.txt")


def send_command(command: str):
    """Write command to file for Webots to read."""
    with open(COMMAND_FILE, 'w') as f:
        f.write(command)
    print(f"[SENT] {command}")


def main():
    print("=" * 60)
    print("  SemanticVLN-MCP Natural Language Command Interface")
    print("=" * 60)
    print("\nType natural language commands to control the robot.")
    print("Examples:")
    print("  'Navigate to the kitchen'")
    print("  'Find the person'")
    print("  'Go to the living room'")
    print("  'I'm hungry' (implicit: go to kitchen)")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            command = input("Command> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            send_command(command)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
