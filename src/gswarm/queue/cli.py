"""
CLI for GSwarm Task Queue Manager
"""

import argparse
import sys
from datetime import datetime
from typing import List
import json

from gswarm.queue.manager import TaskQueueManager, Task


def create_sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        uuid=None,  # Will be auto-generated
        datetime=datetime.now(),
        model_name="sample_model",
        input={"data": "sample input"},
        tag="sample_tag",
        dependencies=set(),
        timeout=30.0,
        others={"priority": 1}
    )


def main():
    parser = argparse.ArgumentParser(description="GSwarm Task Queue Manager CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add task command
    add_parser = subparsers.add_parser('add', help='Add a task to a queue')
    add_parser.add_argument('device_id', help='Device ID')
    add_parser.add_argument('--model', required=True, help='Model name')
    add_parser.add_argument('--input', required=True, help='Input data (JSON)')
    add_parser.add_argument('--tag', default='', help='Task tag')
    add_parser.add_argument('--deps', nargs='*', default=[], help='Dependencies')
    add_parser.add_argument('--timeout', type=float, default=30.0, help='Timeout in seconds')
    add_parser.add_argument('--others', default='{}', help='Other parameters (JSON)')
    
    # Remove task command
    remove_parser = subparsers.add_parser('remove', help='Remove a task from a queue')
    remove_parser.add_argument('device_id', help='Device ID')
    remove_parser.add_argument('task_uuid', help='Task UUID')
    
    # List tasks command
    list_parser = subparsers.add_parser('list', help='List tasks in a queue')
    list_parser.add_argument('device_id', help='Device ID')
    
    # Get task command
    get_parser = subparsers.add_parser('get', help='Get a specific task')
    get_parser.add_argument('device_id', help='Device ID')
    get_parser.add_argument('task_uuid', help='Task UUID')
    
    # Queue size command
    size_parser = subparsers.add_parser('size', help='Get queue size')
    size_parser.add_argument('device_id', help='Device ID')
    
    # List devices command
    subparsers.add_parser('devices', help='List all devices with queues')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = TaskQueueManager()
    
    if args.command == 'add':
        try:
            input_data = json.loads(args.input)
            others_data = json.loads(args.others)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return 1
            
        task = Task(
            uuid=None,
            datetime=datetime.now(),
            model_name=args.model,
            input=input_data,
            tag=args.tag,
            dependencies=set(args.deps),
            timeout=args.timeout,
            others=others_data
        )
        
        manager.add_task(args.device_id, task)
        print(f"Added task {task.uuid} to device {args.device_id}")
        
    elif args.command == 'remove':
        task = manager.remove_task(args.device_id, args.task_uuid)
        if task:
            print(f"Removed task {task.uuid} from device {args.device_id}")
        else:
            print(f"Task {args.task_uuid} not found in device {args.device_id}")
            
    elif args.command == 'list':
        tasks = manager.get_all_tasks(args.device_id)
        if not tasks:
            print(f"No tasks in queue for device {args.device_id}")
            return
            
        print(f"Tasks in queue for device {args.device_id}:")
        for task in tasks:
            print(f"  {task.uuid}: {task.model_name} ({task.tag})")
            
    elif args.command == 'get':
        task = manager.get_task(args.device_id, args.task_uuid)
        if task:
            print(f"Task {task.uuid}:")
            print(f"  Model: {task.model_name}")
            print(f"  Input: {task.input}")
            print(f"  Tag: {task.tag}")
            print(f"  Dependencies: {list(task.dependencies)}")
            print(f"  Timeout: {task.timeout}")
            print(f"  Others: {task.others}")
        else:
            print(f"Task {args.task_uuid} not found in device {args.device_id}")
            
    elif args.command == 'size':
        size = manager.get_queue_size(args.device_id)
        print(f"Queue size for device {args.device_id}: {size}")
        
    elif args.command == 'devices':
        devices = manager.get_all_device_ids()
        if not devices:
            print("No devices with queues")
            return
            
        print("Devices with queues:")
        for device in devices:
            print(f"  {device}")


if __name__ == "__main__":
    sys.exit(main())