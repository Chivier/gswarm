# GSwarm Task Queue Manager

The Task Queue Manager provides a queue implementation for managing tasks assigned to GPU devices in the GSwarm scheduling system.

## Overview

Each GPU device has its own task queue. Tasks are represented by the `Task` dataclass which contains:
- `uuid`: Unique identifier for the task
- `datetime`: Timestamp when the task was created
- `model_name`: Name of the model to execute
- `input`: Input data for the model
- `tag`: Optional tag for categorizing tasks
- `dependencies`: Set of task UUIDs that must complete before this task can run
- `timeout`: Maximum time allowed for task execution
- `others`: Additional metadata

## Usage

### Basic Usage

```python
from gswarm.queue.manager import TaskQueueManager, Task
from datetime import datetime

# Create a queue manager
manager = TaskQueueManager()

# Create a task
task = Task(
    uuid=None,  # Auto-generated if None
    datetime=datetime.now(),
    model_name="my_model",
    input={"data": "input_data"},
    tag="my_tag",
    dependencies=set(),
    timeout=30.0,
    others={"priority": 1}
)

# Add task to a GPU queue
manager.add_task("cuda:0", task)

# Get next task from queue
next_task = manager.get_next_task("cuda:0")

# Remove a task
removed_task = manager.remove_task("cuda:0", task.uuid)
```

### Sorting Tasks

Tasks can be sorted in a queue using a custom value function:

```python
def priority_function(tasks, device_id, **kwargs):
    # Example function that prioritizes by timeout
    return tasks[0].timeout

# Sort tasks in queue
manager.sort_queue("cuda:0", priority_function)
```

### CLI Usage

The queue manager also provides a CLI for testing:

```bash
# Add a task
python -m gswarm.queue.cli add cuda:0 --model my_model --input '{"data": "test"}' --timeout 30

# List tasks
python -m gswarm.queue.cli list cuda:0

# Get a specific task
python -m gswarm.queue.cli get cuda:0 <task_uuid>

# Remove a task
python -m gswarm.queue.cli remove cuda:0 <task_uuid>
```

## Integration with Schedulers

The queue manager is designed to work with the GSwarm scheduler system. Each scheduler can use the queue manager to:
1. Add new tasks to GPU queues
2. Retrieve tasks for execution
3. Sort tasks based on scheduling policies
4. Track task completion and dependencies

See `examples/queue_example.py` for a complete example of integration with schedulers.