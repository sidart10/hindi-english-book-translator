#!/usr/bin/env python3
import json
from collections import defaultdict

# Load tasks
with open('.taskmaster/tasks/tasks.json', 'r') as f:
    data = json.load(f)
    tasks = data['tags']['master']['tasks']

# Group tasks by priority
priority_groups = defaultdict(list)
priority_order = ['high', 'medium', 'low']

for task in tasks:
    priority = task.get('priority', 'medium')
    priority_groups[priority].append(task)

# Display tasks by priority
print("=" * 80)
print("TASKS ORGANIZED BY PRIORITY")
print("=" * 80)
print()

for priority in priority_order:
    if priority in priority_groups:
        print(f"🔴 {priority.upper()} PRIORITY ({len(priority_groups[priority])} tasks)")
        print("-" * 40)
        
        # Sort by dependencies (tasks with fewer deps come first) and then by ID
        sorted_tasks = sorted(priority_groups[priority], 
                            key=lambda x: (len(x['dependencies']), x['id']))
        
        for task in sorted_tasks:
            status_icon = "✅" if task['status'] == 'done' else "⏳"
            deps_str = f"(depends on: {task['dependencies']})" if task['dependencies'] else "(no dependencies)"
            
            print(f"{status_icon} Task {task['id']:2d}: {task['title']}")
            print(f"           Status: {task['status']} | Dependencies: {deps_str}")
            print()
        
        print()

# Show summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

# Count by status
status_counts = defaultdict(int)
for task in tasks:
    status_counts[task['status']] += 1

print(f"Total tasks: {len(tasks)}")
print(f"Completed: {status_counts.get('done', 0)}")
print(f"Pending: {status_counts.get('pending', 0)}")
print()

# Show tasks ready to start (pending with all dependencies done)
print("TASKS READY TO START (pending with dependencies satisfied):")
print("-" * 40)

completed_ids = {task['id'] for task in tasks if task['status'] == 'done'}
ready_tasks = []

for task in tasks:
    if task['status'] == 'pending':
        # Check if all dependencies are done
        if all(dep in completed_ids for dep in task['dependencies']):
            ready_tasks.append(task)

# Sort by priority and ID
priority_rank = {'high': 0, 'medium': 1, 'low': 2}
ready_tasks.sort(key=lambda x: (priority_rank.get(x['priority'], 1), x['id']))

for task in ready_tasks:
    print(f"  🚀 Task {task['id']:2d}: {task['title']} (Priority: {task['priority']})")

if not ready_tasks:
    print("  None - all tasks are either done or waiting for dependencies") 