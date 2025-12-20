#!/usr/bin/env python3
"""
Demo script for the Enhanced Service Manager and distributed perception services.

This script demonstrates:
1. Starting the service manager
2. Registering FastSAM and Grasp services
3. Using the orchestrated perception pipeline
4. Dynamic prompt updates
5. Service health monitoring
"""

import time
import threading
import logging
from typing import Dict, Any
from pathlib import Path

# Import our services with absolute imports
from robosuite_franka_manipulation.services.service_manager import EnhancedServiceManager, ServiceClient
from robosuite_franka_manipulation.services.fastsam_service import FastSAMService
from robosuite_franka_manipulation.services.grasp_service import GraspService

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def start_service_manager() -> EnhancedServiceManager:
    """Start the enhanced service manager."""
    logger.info("Starting Enhanced Service Manager...")
    manager = EnhancedServiceManager(host='localhost', port=5555)
    manager.start()
    time.sleep(1)  # Allow time to start
    return manager


def start_fastsam_service():
    """Start the FastSAM service."""
    logger.info("FastSAM Service requires robosuite environment - skipping for demo")
    logger.info("In a real scenario, create env = make_dual_arm_env() and pass to FastSAMService")
    return None


def start_grasp_service():
    """Start the Grasp service."""
    logger.info("Grasp Service requires model files - skipping for demo")
    logger.info("In a real scenario, provide path to GraspGenModels checkpoints")
    return None


def demo_service_client() -> None:
    """Demonstrate using the ServiceClient for various operations."""
    logger.info("=== Service Client Demo ===")

    client = ServiceClient(manager_host='localhost', manager_port=5555)

    # Wait for services to be ready
    time.sleep(3)

    # Check service status
    logger.info("Checking service status...")
    status = client.get_service_status()
    logger.info(f"Service Status: {status}")

    # Set detection prompt
    logger.info("Setting detection prompt to 'red cup'...")
    result = client.set_detection_prompt("red cup")
    logger.info(f"Prompt update result: {result}")

    # Get perception pipeline results
    logger.info("Getting perception pipeline results...")
    pipeline_result = client.get_perception_pipeline(num_grasps=3)
    logger.info(f"Pipeline Result: {pipeline_result}")

    # Direct service calls
    logger.info("Making direct service calls...")

    # Call FastSAM directly
    fastsam_result = client.call('fastsam_service', {'command': 'GET_STATUS'})
    logger.info(f"FastSAM Status: {fastsam_result}")

    # Call Grasp service directly
    grasp_result = client.call('grasp_service', {'command': 'GET_STATUS'})
    logger.info(f"Grasp Status: {grasp_result}")


def demo_orchestrated_pipeline() -> None:
    """Demonstrate the orchestrated perception pipeline."""
    logger.info("=== Orchestrated Pipeline Demo ===")

    client = ServiceClient(manager_host='localhost', manager_port=5555)

    # Test different prompts
    prompts = ["red cup", "blue bottle", "yellow banana"]

    for prompt in prompts:
        logger.info(f"\n--- Testing with prompt: '{prompt}' ---")

        # Set prompt
        client.set_detection_prompt(prompt)
        time.sleep(1)  # Allow time for prompt to take effect

        # Get pipeline results
        result = client.get_perception_pipeline(num_grasps=5)

        if result['status'] == 'success':
            pipeline_results = result.get('pipeline_results', [])
            logger.info(f"Found {len(pipeline_results)} detections")

            for i, res in enumerate(pipeline_results):
                detection = res.get('detection', {})
                grasps = res.get('grasps', [])
                logger.info(f"  Detection {i+1}: {detection.get('class', 'unknown')} "
                          f"with {len(grasps)} grasps")
        else:
            logger.warning(f"Pipeline failed: {result.get('message', 'Unknown error')}")

        time.sleep(2)  # Wait between tests


def main():
    """Main demo function."""
    logger.info("Starting Distributed Perception Services Demo")

    # Start service manager
    manager = start_service_manager()

    # Start services
    fastsam = start_fastsam_service()
    grasp = start_grasp_service()

    try:
        # Wait for everything to initialize
        logger.info("Waiting for services to initialize...")
        time.sleep(5)

        # Run client demo
        demo_service_client()

        # Run orchestrated pipeline demo
        demo_orchestrated_pipeline()

        # Keep services running for manual testing
        logger.info("\n=== Services running - you can now test manually ===")
        logger.info("Use ServiceClient to interact with services")
        logger.info("Press Ctrl+C to stop all services")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping demo...")

    finally:
        # Cleanup
        logger.info("Stopping services...")
        if 'fastsam' in locals():
            fastsam.stop()
        if 'grasp' in locals():
            grasp.stop()
        if 'manager' in locals():
            manager.stop()

        logger.info("Demo completed")


if __name__ == "__main__":
    main()