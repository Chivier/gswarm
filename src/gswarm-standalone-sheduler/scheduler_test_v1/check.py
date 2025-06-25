#!/usr/bin/env python3
"""
JSON GPU Execution Checker
Validates that GPU executions don't overlap in time.
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse


class GPUExecutionChecker:
    def __init__(self, json_file: str):
        """Initialize the checker with a JSON execution log file."""
        self.json_file = json_file
        self.data = None
        self.executions_by_gpu = defaultdict(list)
        self.conflicts = []

    def load_data(self) -> bool:
        """Load and parse the JSON file."""
        try:
            with open(self.json_file, "r") as f:
                self.data = json.load(f)
            print(f"✓ Successfully loaded {self.json_file}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: File {self.json_file} not found")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON format - {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return False

    def validate_structure(self) -> bool:
        """Validate that the JSON has the expected structure."""
        if not isinstance(self.data, dict):
            print("✗ Error: Root should be a JSON object")
            return False

        if "executions" not in self.data:
            print("✗ Error: Missing 'executions' field")
            return False

        if not isinstance(self.data["executions"], list):
            print("✗ Error: 'executions' should be a list")
            return False

        print(f"✓ JSON structure is valid")
        return True

    def group_executions_by_gpu(self):
        """Group executions by GPU ID and sort by start time."""
        for execution in self.data["executions"]:
            # Validate required fields
            required_fields = ["gpu_id", "start_time", "end_time", "request_id"]
            for field in required_fields:
                if field not in execution:
                    print(f"✗ Warning: Execution missing field '{field}': {execution}")
                    continue

            gpu_id = execution["gpu_id"]
            self.executions_by_gpu[gpu_id].append(execution)

        # Sort executions by start time for each GPU
        for gpu_id in self.executions_by_gpu:
            self.executions_by_gpu[gpu_id].sort(key=lambda x: x["start_time"])

        print(f"✓ Grouped {len(self.data['executions'])} executions across {len(self.executions_by_gpu)} GPUs")

    def check_overlaps(self) -> bool:
        """Check for overlapping executions on each GPU."""
        self.conflicts = []
        total_overlaps = 0

        for gpu_id, executions in self.executions_by_gpu.items():
            gpu_overlaps = self._check_gpu_overlaps(gpu_id, executions)
            total_overlaps += len(gpu_overlaps)
            self.conflicts.extend(gpu_overlaps)

        if total_overlaps == 0:
            print(f"✓ No overlapping executions found across all GPUs")
            return True
        else:
            print(f"✗ Found {total_overlaps} overlapping execution pairs")
            return False

    def _check_gpu_overlaps(self, gpu_id: int, executions: List[Dict]) -> List[Dict]:
        """Check for overlaps within a single GPU's executions."""
        overlaps = []

        for i in range(len(executions)):
            for j in range(i + 1, len(executions)):
                exec1 = executions[i]
                exec2 = executions[j]

                # Check if they overlap
                if self._executions_overlap(exec1, exec2):
                    overlap_info = {
                        "gpu_id": gpu_id,
                        "execution1": exec1,
                        "execution2": exec2,
                        "overlap_start": max(exec1["start_time"], exec2["start_time"]),
                        "overlap_end": min(exec1["end_time"], exec2["end_time"]),
                    }
                    overlaps.append(overlap_info)

        return overlaps

    def _executions_overlap(self, exec1: Dict, exec2: Dict) -> bool:
        """Check if two executions overlap in time."""
        start1, end1 = exec1["start_time"], exec1["end_time"]
        start2, end2 = exec2["start_time"], exec2["end_time"]

        # Two intervals overlap if: start1 < end2 AND start2 < end1
        return start1 < end2 and start2 < end1

    def print_summary(self):
        """Print a summary of the validation results."""
        if not self.data:
            return

        print("\n" + "=" * 80)
        print("GPU EXECUTION VALIDATION SUMMARY")
        print("=" * 80)

        # Print overall statistics
        summary = self.data.get("summary", {})
        total_requests = summary.get("total_requests", "N/A")
        total_nodes = summary.get("total_nodes_executed", "N/A")
        makespan = summary.get("makespan", "N/A")
        gpus = summary.get("gpus", [])

        print(f"Total Requests: {total_requests}")
        print(f"Total Nodes Executed: {total_nodes}")
        print(f"Makespan: {makespan}")
        print(f"Available GPUs: {len(gpus)} ({min(gpus) if gpus else 'N/A'}-{max(gpus) if gpus else 'N/A'})")
        print(f"Total Executions: {len(self.data['executions'])}")

        # Print GPU utilization
        print(f"\nGPU Utilization:")
        for gpu_id in sorted(self.executions_by_gpu.keys()):
            count = len(self.executions_by_gpu[gpu_id])
            print(f"  GPU {gpu_id:2d}: {count:4d} executions")

        # Print conflicts if any
        if self.conflicts:
            print(f"\n✗ CONFLICTS DETECTED: {len(self.conflicts)} overlapping execution pairs")
            self.print_conflicts()
        else:
            print(f"\n✓ NO CONFLICTS: All GPU executions are properly scheduled")

    def print_conflicts(self):
        """Print detailed information about conflicts."""
        print("\nDETAILED CONFLICT REPORT:")
        print("-" * 80)

        for i, conflict in enumerate(self.conflicts, 1):
            gpu_id = conflict["gpu_id"]
            exec1 = conflict["execution1"]
            exec2 = conflict["execution2"]
            overlap_start = conflict["overlap_start"]
            overlap_end = conflict["overlap_end"]
            overlap_duration = overlap_end - overlap_start

            print(f"\nConflict #{i} - GPU {gpu_id}:")
            print(f"  Execution 1: {exec1['request_id']} ({exec1['start_time']:.2f} - {exec1['end_time']:.2f})")
            print(f"  Execution 2: {exec2['request_id']} ({exec2['start_time']:.2f} - {exec2['end_time']:.2f})")
            print(f"  Overlap: {overlap_start:.2f} - {overlap_end:.2f} (duration: {overlap_duration:.2f})")
            print(f"  Models: {exec1.get('model_name', 'N/A')} & {exec2.get('model_name', 'N/A')}")

    def save_conflicts_report(self, output_file: str = None):
        """Save conflicts to a JSON file."""
        if not output_file:
            output_file = self.json_file.replace(".json", "_conflicts_report.json")

        report = {
            "source_file": self.json_file,
            "total_conflicts": len(self.conflicts),
            "conflicts": self.conflicts,
            "summary": {
                "total_executions": len(self.data["executions"]),
                "total_gpus_used": len(self.executions_by_gpu),
                "validation_passed": len(self.conflicts) == 0,
            },
        }

        try:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"✓ Conflicts report saved to: {output_file}")
        except Exception as e:
            print(f"✗ Error saving conflicts report: {e}")

    def validate(self, save_report: bool = False) -> bool:
        """Run the complete validation process."""
        print(f"Starting GPU execution validation for: {self.json_file}")
        print("=" * 80)

        # Load and validate data
        if not self.load_data():
            return False

        if not self.validate_structure():
            return False

        # Group executions and check for overlaps
        self.group_executions_by_gpu()
        is_valid = self.check_overlaps()

        # Print summary
        self.print_summary()

        # Save report if requested
        if save_report:
            self.save_conflicts_report()

        return is_valid


def main():
    parser = argparse.ArgumentParser(description="Check GPU execution schedules for overlaps")
    parser.add_argument(
        "json_file",
        nargs="?",
        default="baseline_execution_log.json",
        help="JSON file to check (default: baseline_execution_log.json)",
    )
    parser.add_argument("--save-report", action="store_true", help="Save conflicts report to a file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show summary results")

    args = parser.parse_args()

    if args.quiet:
        # Redirect detailed output for quiet mode
        import os

        devnull = open(os.devnull, "w")
        sys.stdout = devnull

    checker = GPUExecutionChecker(args.json_file)
    is_valid = checker.validate(save_report=args.save_report)

    if args.quiet:
        sys.stdout = sys.__stdout__
        if is_valid:
            print("✓ PASS: No GPU execution overlaps detected")
        else:
            print(f"✗ FAIL: {len(checker.conflicts)} GPU execution overlaps detected")

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
