#!/usr/bin/env python3
"""
Performance and stress tests for GSwarm
"""

import unittest
import sys
import os
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from gswarm.model import CostModel
from gswarm.predictor import predictor, Predictor
from gswarm.scheduler import (
    BaselineScheduler,
    OfflineScheduler,
    OnlineScheduler,
    ModelInfo,
    Workflow,
    WorkflowNode,
    WorkflowEdge,
    Request
)


class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.num_devices = 16  # Simulate 16 GPUs
        self.devices = [f"node{i//4}:cuda:{i%4}" for i in range(self.num_devices)]
        
        # Create a variety of models
        self.models = {
            "gpt-4": ModelInfo(name="gpt-4", memory_gb=16.0, gpus_required=1, 
                             load_time_seconds=5.0, tokens_per_second=50.0),
            "llama-70b": ModelInfo(name="llama-70b", memory_gb=40.0, gpus_required=2,
                                 load_time_seconds=10.0, tokens_per_second=20.0),
            "stable-diffusion-xl": ModelInfo(name="stable-diffusion-xl", memory_gb=10.0, 
                                           gpus_required=1, load_time_seconds=4.0, 
                                           inference_time_mean=3.0),
            "whisper-large": ModelInfo(name="whisper-large", memory_gb=3.0, gpus_required=1,
                                     load_time_seconds=2.0, inference_time_mean=1.5),
            "llama-7b": ModelInfo(name="llama-7b", memory_gb=13.0, gpus_required=1,
                                load_time_seconds=3.0, tokens_per_second=40.0)
        }
    
    def test_predictor_throughput(self):
        """Test predictor throughput with many concurrent requests"""
        print("\n=== Testing Predictor Throughput ===")
        
        # Create multiple predictor instances
        predictors = [Predictor() for _ in range(4)]
        
        # Test data
        test_requests = []
        for i in range(1000):
            if i % 2 == 0:
                # LLM request
                test_requests.append({
                    "model_name": "gpt-4" if i % 4 == 0 else "llama-7b",
                    "device": self.devices[i % len(self.devices)],
                    "inputs": {"prompt": "Test prompt " * (i % 10 + 1)}
                })
            else:
                # SD request
                test_requests.append({
                    "model_name": "stable-diffusion-xl",
                    "device": self.devices[i % len(self.devices)],
                    "inputs": {"height": 512 + (i % 3) * 256, "width": 512 + (i % 3) * 256}
                })
        
        # Measure single-threaded performance
        start_time = time.time()
        single_thread_results = []
        for req in test_requests:
            pred = predictors[0].predict(req["model_name"], req["device"], req["inputs"])
            single_thread_results.append(pred)
        single_thread_time = time.time() - start_time
        
        print(f"Single-threaded: {len(test_requests)} predictions in {single_thread_time:.2f}s")
        print(f"Throughput: {len(test_requests)/single_thread_time:.2f} predictions/second")
        
        # Measure multi-threaded performance
        def predict_batch(predictor_idx, requests):
            predictor = predictors[predictor_idx]
            results = []
            for req in requests:
                pred = predictor.predict(req["model_name"], req["device"], req["inputs"])
                results.append(pred)
            return results
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            chunk_size = len(test_requests) // 4
            for i in range(4):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < 3 else len(test_requests)
                chunk = test_requests[start_idx:end_idx]
                future = executor.submit(predict_batch, i, chunk)
                futures.append(future)
            
            multi_thread_results = []
            for future in as_completed(futures):
                multi_thread_results.extend(future.result())
        
        multi_thread_time = time.time() - start_time
        
        print(f"\nMulti-threaded (4 threads): {len(test_requests)} predictions in {multi_thread_time:.2f}s")
        print(f"Throughput: {len(test_requests)/multi_thread_time:.2f} predictions/second")
        print(f"Speedup: {single_thread_time/multi_thread_time:.2f}x")
        
        # Verify results are consistent
        self.assertEqual(len(single_thread_results), len(multi_thread_results))
    
    def test_scheduler_scalability(self):
        """Test scheduler performance with large number of requests"""
        print("\n=== Testing Scheduler Scalability ===")
        
        # Create a complex workflow
        nodes = []
        edges = []
        
        # Create a DAG with multiple stages
        stages = 4
        nodes_per_stage = 3
        
        for stage in range(stages):
            for node in range(nodes_per_stage):
                node_id = f"stage{stage}_node{node}"
                model = list(self.models.keys())[node % len(self.models)]
                nodes.append(WorkflowNode(
                    id=node_id,
                    model=model,
                    inputs=[f"input_{stage}_{node}"],
                    outputs=[f"output_{stage}_{node}"]
                ))
                
                # Connect to previous stage
                if stage > 0:
                    prev_node_id = f"stage{stage-1}_node{node}"
                    edges.append(WorkflowEdge(from_node=prev_node_id, to_node=node_id))
        
        workflow = Workflow(id="complex_workflow", name="Complex", nodes=nodes, edges=edges)
        
        # Test different schedulers
        schedulers = [
            ("Baseline", BaselineScheduler(devices=self.devices, models=self.models, simulate=True)),
            ("Offline", OfflineScheduler(devices=self.devices, models=self.models)),
            ("Online", OnlineScheduler(devices=self.devices, models=self.models))
        ]
        
        num_requests = 100
        
        for scheduler_name, scheduler in schedulers:
            scheduler.add_workflow(workflow)
            
            # Create requests
            requests = []
            for i in range(num_requests):
                requests.append(Request(
                    id=f"req_{i}",
                    workflow_id="complex_workflow",
                    arrival_time=i * 0.1,  # Arrival every 0.1 seconds
                    priority=i % 5
                ))
            
            # Measure scheduling time
            start_time = time.time()
            
            if scheduler_name == "Offline":
                # Batch scheduling
                tasks = scheduler.schedule(requests)
                scheduling_time = time.time() - start_time
                print(f"\n{scheduler_name} Scheduler:")
                print(f"  Scheduled {len(tasks)} tasks in {scheduling_time:.3f}s")
                print(f"  Tasks per second: {len(tasks)/scheduling_time:.2f}")
            else:
                # Online scheduling
                task_count = 0
                for req in requests:
                    scheduler.add_request(req)
                    while True:
                        task = scheduler.get_next_task()
                        if task is None:
                            break
                        task_count += 1
                        scheduler.complete_task(task)
                
                scheduling_time = time.time() - start_time
                print(f"\n{scheduler_name} Scheduler:")
                print(f"  Processed {task_count} tasks in {scheduling_time:.3f}s")
                print(f"  Tasks per second: {task_count/scheduling_time:.2f}")
            
            # Get metrics
            metrics = scheduler.get_metrics()
            if hasattr(metrics, 'model_switch_count'):
                print(f"  Model switches: {metrics.model_switch_count}")
    
    def test_device_scaling(self):
        """Test system performance with increasing number of devices"""
        print("\n=== Testing Device Scaling ===")
        
        # Simple workflow for testing
        nodes = [
            WorkflowNode(id="process", model="gpt-4", inputs=["input"], outputs=["output"])
        ]
        workflow = Workflow(id="simple", name="Simple", nodes=nodes, edges=[])
        
        device_counts = [4, 8, 16, 32, 64]
        results = []
        
        for device_count in device_counts:
            # Create devices
            devices = [f"node{i//4}:cuda:{i%4}" for i in range(device_count)]
            
            # Create scheduler
            scheduler = BaselineScheduler(devices=devices, models=self.models, simulate=True)
            scheduler.add_workflow(workflow)
            
            # Create requests
            num_requests = device_count * 10
            requests = [Request(id=f"req_{i}", workflow_id="simple", arrival_time=0.0) 
                       for i in range(num_requests)]
            
            # Measure processing time
            start_time = time.time()
            
            for req in requests:
                scheduler.add_request(req)
            
            tasks_processed = 0
            while tasks_processed < num_requests:
                task = scheduler.get_next_task()
                if task:
                    scheduler.complete_task(task)
                    tasks_processed += 1
                else:
                    scheduler.advance_time(scheduler.current_time + 0.1)
            
            processing_time = time.time() - start_time
            throughput = num_requests / processing_time
            
            results.append({
                "devices": device_count,
                "requests": num_requests,
                "time": processing_time,
                "throughput": throughput
            })
            
            print(f"\nDevices: {device_count}")
            print(f"  Requests: {num_requests}")
            print(f"  Time: {processing_time:.2f}s")
            print(f"  Throughput: {throughput:.2f} requests/second")
        
        # Check scaling efficiency
        base_throughput = results[0]["throughput"]
        for i, result in enumerate(results[1:], 1):
            scaling_factor = result["devices"] / results[0]["devices"]
            actual_speedup = result["throughput"] / base_throughput
            efficiency = actual_speedup / scaling_factor * 100
            print(f"\n{result['devices']} devices: {efficiency:.1f}% scaling efficiency")
    
    def test_memory_efficiency(self):
        """Test memory usage patterns"""
        print("\n=== Testing Memory Efficiency ===")
        
        # Create many cost model instances
        import gc
        import tracemalloc
        
        tracemalloc.start()
        
        # Baseline memory
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Create many predictors
        predictors = []
        for i in range(100):
            p = Predictor()
            predictors.append(p)
            # Make some predictions
            for j in range(10):
                p.predict("gpt-4", f"node{i}:cuda:{j%4}", {"prompt": "Test"})
        
        # Check memory after creation
        current = tracemalloc.get_traced_memory()[0]
        memory_used = (current - baseline) / 1024 / 1024  # MB
        
        print(f"\nMemory used by 100 predictors: {memory_used:.2f} MB")
        print(f"Average per predictor: {memory_used/100:.2f} MB")
        
        # Clear and check memory is released
        predictors.clear()
        gc.collect()
        
        after_clear = tracemalloc.get_traced_memory()[0]
        memory_released = (current - after_clear) / 1024 / 1024
        
        print(f"Memory released: {memory_released:.2f} MB")
        print(f"Release efficiency: {memory_released/memory_used*100:.1f}%")
        
        tracemalloc.stop()
    
    def test_concurrent_updates(self):
        """Test concurrent model updates"""
        print("\n=== Testing Concurrent Updates ===")
        
        cost_model = CostModel()
        
        # Function to perform updates
        def update_model(thread_id, num_updates):
            for i in range(num_updates):
                device = f"thread{thread_id}:cuda:{i%4}"
                if i % 2 == 0:
                    cost_model.update(
                        "gpt-4", device,
                        {"prompt": f"Test {i}", "output": f"Response {i}"},
                        actual_time=0.5 + (i % 10) * 0.1
                    )
                else:
                    cost_model.update(
                        "stable-diffusion", device,
                        {"height": 512, "width": 512, "processing_time": 2.0 + (i % 5) * 0.2},
                        actual_time=2.0 + (i % 5) * 0.2
                    )
        
        # Run concurrent updates
        num_threads = 8
        updates_per_thread = 100
        
        start_time = time.time()
        
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=update_model, args=(i, updates_per_thread))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        update_time = time.time() - start_time
        total_updates = num_threads * updates_per_thread
        
        print(f"\nConcurrent updates: {total_updates} in {update_time:.2f}s")
        print(f"Update rate: {total_updates/update_time:.2f} updates/second")
        
        # Verify model still works
        test_time = cost_model.predict("gpt-4", "test:cuda:0", {"prompt": "Final test"})
        self.assertGreater(test_time, 0)
        print(f"Model still functional after concurrent updates: {test_time:.2f}s")


if __name__ == "__main__":
    unittest.main()