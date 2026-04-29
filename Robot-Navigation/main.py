"""
SemanticVLN-MCP: Main Entry Point
==================================

Semantic Vision-Language Navigation with Multi-Agent Coordination

Usage:
    python main.py                      # Start with default settings
    python main.py --device cpu         # CPU-only mode
    python main.py --command "Go to kitchen"  # Execute single command

Author: SemanticVLN-MCP Team
"""

import argparse
import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_vln_mcp.mcp.orchestrator import MCPOrchestrator, RobotState


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SemanticVLN-MCP: Vision-Language Navigation System"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for neural network inference (default: cuda)"
    )
    
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Single navigation command to execute"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (command line input)"
    )
    
    parser.add_argument(
        "--webots",
        action="store_true",
        help="Run with Webots integration"
    )
    
    parser.add_argument(
        "--web-dashboard",
        action="store_true",
        help="Start web dashboard"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Web dashboard port (default: 5000)"
    )
    
    return parser.parse_args()


async def interactive_mode(orchestrator: MCPOrchestrator):
    """Run in interactive command-line mode."""
    print("\n" + "="*60)
    print("SemanticVLN-MCP Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  - Type any navigation instruction")
    print("  - 'status' - Show current status")
    print("  - 'stop' - Stop navigation")
    print("  - 'quit' - Exit")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("Command> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("Exiting...")
                break
            
            if user_input.lower() == "status":
                status = orchestrator.get_status()
                print("\n--- Status ---")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                print("--------------\n")
                continue
            
            if user_input.lower() == "stop":
                orchestrator.context.state = RobotState.STOPPED
                print("Stopped.")
                continue
            
            # Process navigation command
            print(f"\nProcessing: \"{user_input}\"")
            await orchestrator.process_instruction(user_input)
            
            # Show result
            status = orchestrator.get_status()
            print(f"State: {status['state']}")
            if status['goal_position']:
                print(f"Goal: {status['goal_position']}")
            if status['path_waypoints'] > 0:
                print(f"Path: {status['path_waypoints']} waypoints")
            print()
            
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


async def single_command_mode(orchestrator: MCPOrchestrator, command: str):
    """Execute a single command."""
    print(f"Executing: \"{command}\"")
    await orchestrator.process_instruction(command)
    
    status = orchestrator.get_status()
    print("\nResult:")
    print(f"  State: {status['state']}")
    print(f"  Goal: {status['goal_position']}")
    print(f"  Path waypoints: {status['path_waypoints']}")


def run_webots_controller():
    """Run the Webots controller."""
    from semantic_vln_mcp.webots.controllers.semantic_vln_controller.semantic_vln_controller import SemanticVLNController
    
    controller = SemanticVLNController()
    controller.run()


def run_web_dashboard(orchestrator: MCPOrchestrator, port: int):
    """Start web dashboard."""
    try:
        from semantic_vln_mcp.interfaces.web_dashboard.app import create_app
        
        app = create_app(orchestrator)
        print(f"Starting web dashboard on http://localhost:{port}")
        app.run(host="0.0.0.0", port=port, debug=False)
    except ImportError:
        print("Web dashboard not available. Install flask and flask-socketio.")


async def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("  SemanticVLN-MCP")
    print("  Vision-Language Navigation with Multi-Agent Coordination")
    print("="*60)
    print(f"  Device: {args.device}")
    print("="*60 + "\n")
    
    # Webots mode
    if args.webots:
        run_webots_controller()
        return
    
    # Initialize orchestrator
    print("Initializing system...")
    orchestrator = MCPOrchestrator(device=args.device)
    
    # Web dashboard mode
    if args.web_dashboard:
        run_web_dashboard(orchestrator, args.port)
        return
    
    # Single command mode
    if args.command:
        await single_command_mode(orchestrator, args.command)
        return
    
    # Interactive mode (default)
    await interactive_mode(orchestrator)


if __name__ == "__main__":
    asyncio.run(main())
