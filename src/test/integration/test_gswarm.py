#!/usr/bin/env python3
"""
GSwarm Comprehensive Test Script
Tests cost models, predictors, and schedulers with the new device format
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from gswarm.model import CostModel, LLMCostModel, SDCostModel
from gswarm.predictor import Predictor, predictor, predict_inference
from gswarm.scheduler import (
    BaselineScheduler,
    OfflineScheduler,
    OnlineScheduler,
    StaticScheduler,
    ModelInfo,
    Workflow,
    WorkflowNode,
    WorkflowEdge,
    Request
)
from gswarm.utils import parse_device, format_device, normalize_device


class GSwarmTester:
    """Comprehensive tester for GSwarm components"""
    
    def __init__(self, host_url: str = "http://localhost:8080", verbose: bool = False):
        self.host_url = host_url
        self.verbose = verbose
        self.results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color_map = {
            "INFO": "\033[0;32m",    # Green
            "WARNING": "\033[1;33m", # Yellow
            "ERROR": "\033[0;31m",   # Red
            "DEBUG": "\033[0;34m"    # Blue
        }
        color = color_map.get(level, "")
        reset = "\033[0m"
        
        print(f"{color}[{timestamp}] [{level}] {message}{reset}")
    
    def test_device_utilities(self) -> bool:
        """Test device parsing and formatting utilities"""
        self.log("Testing device utilities...")
        
        try:
            # Test parsing new format
            device_info = parse_device("node1:cuda:0")
            assert device_info.client == "node1"
            assert device_info.device_type == "cuda"
            assert device_info.device_id == 0
            
            # Test parsing legacy format
            device_info = parse_device("cuda:1")
            assert device_info.client == "localhost"
            assert device_info.device_type == "cuda"
            assert device_info.device_id == 1
            
            # Test formatting
            device_str = format_device("worker2", "cuda", 3)
            assert device_str == "worker2:cuda:3"
            
            # Test normalization
            normalized = normalize_device("cuda:0")
            assert normalized == "localhost:cuda:0"
            
            self.log("✓ Device utilities test passed", "INFO")
            return True
            
        except Exception as e:
            self.log(f"✗ Device utilities test failed: {e}", "ERROR")
            return False
    
    def test_cost_models(self) -> bool:
        """Test cost prediction models"""
        self.log("Testing cost models...")
        
        try:
            # Test unified CostModel
            cost_model = CostModel()
            
            # Test LLM prediction with new device format
            llm_time = cost_model.predict(
                "gpt-4",
                "node1:cuda:0",
                {"prompt": "Hello world"}
            )
            assert llm_time > 0
            
            # Test SD prediction with new device format
            sd_time = cost_model.predict(
                "stable-diffusion-xl",
                "node2:cuda:1",
                {"height": 1024, "width": 1024}
            )
            assert sd_time > 0
            
            # Test legacy format compatibility
            legacy_time = cost_model.predict(
                "llama-2-7b",
                "cuda:0",
                {"prompt": "Test"}
            )
            assert legacy_time > 0
            
            # Test model update
            cost_model.update(
                "gpt-4",
                "node1:cuda:0",
                {"prompt": "Hi", "output": "Hello"},
                actual_time=0.5
            )
            
            self.log("✓ Cost models test passed", "INFO")
            return True
            
        except Exception as e:
            self.log(f"✗ Cost models test failed: {e}", "ERROR")
            return False
    
    def test_predictor(self) -> bool:
        """Test predictor functionality"""
        self.log("Testing predictor...")
        
        try:
            # Test single prediction
            time1 = predictor.predict(
                "gpt-3.5-turbo",
                "worker1:cuda:0",
                {"prompt": "Explain quantum computing"}
            )
            assert time1 > 0
            
            # Test batch prediction with mixed formats
            requests = [
                {"model_name": "gpt-4", "device": "node1:cuda:0", "inputs": {"prompt": "Q1"}},
                {"model_name": "llama-2", "device": "cuda:1", "inputs": {"prompt": "Q2"}},
                {"model_name": "stable-diffusion", "device": "node3:cuda:0", "inputs": {"height": 512, "width": 512}}
            ]
            times = predictor.batch_predict(requests)
            assert len(times) == 3
            assert all(t > 0 for t in times)
            
            # Test predict_inference function
            result = predict_inference(
                "stable-diffusion-v1-5",
                "worker2:cuda:1",
                {"height": 512, "width": 512, "prompt": "A sunset"}
            )
            assert result.prediction is not None
            
            self.log("✓ Predictor test passed", "INFO")
            return True
            
        except Exception as e:
            self.log(f"✗ Predictor test failed: {e}", "ERROR")
            return False
    
    def test_schedulers(self) -> bool:
        """Test scheduler functionality with new device format"""
        self.log("Testing schedulers...")
        
        try:
            # Define test models
            models = {
                "gpt-4": ModelInfo(
                    name="gpt-4",
                    memory_gb=16.0,
                    gpus_required=1,
                    load_time_seconds=5.0,
                    tokens_per_second=50.0
                ),
                "stable-diffusion": ModelInfo(
                    name="stable-diffusion",
                    memory_gb=8.0,
                    gpus_required=1,
                    load_time_seconds=3.0,
                    inference_time_mean=2.5
                )
            }
            
            # Define test workflow
            nodes = [
                WorkflowNode(id="prompt", model="gpt-4", inputs=["user"], outputs=["text"]),
                WorkflowNode(id="image", model="stable-diffusion", inputs=["text"], outputs=["image"])
            ]
            edges = [WorkflowEdge(from_node="prompt", to_node="image")]
            workflow = Workflow(id="test_workflow", name="Test", nodes=nodes, edges=edges)
            
            # Test baseline scheduler with new device format
            devices = ["node1:cuda:0", "node1:cuda:1", "node2:cuda:0"]
            scheduler = BaselineScheduler(devices=devices, models=models, simulate=True)
            scheduler.add_workflow(workflow)
            
            request = Request(id="req1", workflow_id="test_workflow", arrival_time=0.0)
            scheduler.add_request(request)
            
            task = scheduler.get_next_task()
            assert task is not None
            assert task.device in devices
            
            # Test offline scheduler
            offline_scheduler = OfflineScheduler(devices=devices, models=models)
            offline_scheduler.add_workflow(workflow)
            
            requests = [Request(id=f"req{i}", workflow_id="test_workflow", arrival_time=0.0) for i in range(3)]
            tasks = offline_scheduler.schedule(requests)
            assert len(tasks) > 0
            
            self.log("✓ Schedulers test passed", "INFO")
            return True
            
        except Exception as e:
            self.log(f"✗ Schedulers test failed: {e}", "ERROR")
            return False
    
    def test_host_api(self) -> bool:
        """Test host API endpoints"""
        self.log(f"Testing host API at {self.host_url}...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.host_url}/health", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Health check failed: {response.status_code}")
            
            # Test client list
            response = requests.get(f"{self.host_url}/api/v1/clients", timeout=5)
            if response.status_code == 200:
                clients = response.json().get("clients", [])
                self.log(f"Found {len(clients)} connected clients")
                
                # Display client info with new device format
                for client in clients:
                    devices = client.get("devices", [])
                    self.log(f"  Client: {client['id']}, Devices: {devices}")
            
            self.log("✓ Host API test passed", "INFO")
            return True
            
        except requests.exceptions.ConnectionError:
            self.log("✗ Host API test failed: Cannot connect to host", "WARNING")
            self.log("  Make sure the host is running with: ./start_host.sh", "WARNING")
            return False
        except Exception as e:
            self.log(f"✗ Host API test failed: {e}", "ERROR")
            return False
    
    def test_end_to_end(self) -> bool:
        """Test end-to-end workflow execution"""
        self.log("Testing end-to-end workflow...")
        
        try:
            # Create a test workflow request
            workflow_data = {
                "workflow": {
                    "id": "e2e_test",
                    "name": "End-to-End Test",
                    "nodes": [
                        {
                            "id": "llm_node",
                            "model": "gpt-4",
                            "inputs": {"prompt": "Generate a description"},
                            "device": "node1:cuda:0"  # Specific device assignment
                        },
                        {
                            "id": "sd_node",
                            "model": "stable-diffusion",
                            "inputs": {"height": 512, "width": 512},
                            "device": "node2:cuda:1"  # Different node
                        }
                    ],
                    "edges": [
                        {"from": "llm_node", "to": "sd_node"}
                    ]
                }
            }
            
            # Submit workflow
            response = requests.post(
                f"{self.host_url}/api/v1/workflows/submit",
                json=workflow_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                workflow_id = result.get("workflow_id")
                self.log(f"Workflow submitted: {workflow_id}")
                
                # Check status
                status_response = requests.get(
                    f"{self.host_url}/api/v1/workflows/{workflow_id}/status",
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    self.log(f"Workflow status: {status.get('status')}")
                
                self.log("✓ End-to-end test passed", "INFO")
                return True
            else:
                raise Exception(f"Workflow submission failed: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.log("✗ End-to-end test skipped: Host not available", "WARNING")
            return False
        except Exception as e:
            self.log(f"✗ End-to-end test failed: {e}", "ERROR")
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        self.log("Starting GSwarm comprehensive tests...", "INFO")
        
        tests = [
            ("Device Utilities", self.test_device_utilities),
            ("Cost Models", self.test_cost_models),
            ("Predictor", self.test_predictor),
            ("Schedulers", self.test_schedulers),
            ("Host API", self.test_host_api),
            ("End-to-End", self.test_end_to_end)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n{'='*50}")
            self.log(f"Running: {test_name}")
            self.log(f"{'='*50}")
            
            start_time = time.time()
            try:
                passed = test_func()
                duration = time.time() - start_time
                
                result = {
                    "name": test_name,
                    "passed": passed,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results["tests"].append(result)
                if passed:
                    self.results["passed"] += 1
                else:
                    self.results["failed"] += 1
                    
            except Exception as e:
                self.log(f"Unexpected error in {test_name}: {e}", "ERROR")
                self.results["tests"].append({
                    "name": test_name,
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                self.results["failed"] += 1
        
        self.results["end_time"] = datetime.now().isoformat()
        self.generate_report()
    
    def generate_report(self):
        """Generate test report"""
        self.log(f"\n{'='*60}")
        self.log("TEST SUMMARY")
        self.log(f"{'='*60}")
        
        total_tests = len(self.results["tests"])
        passed = self.results["passed"]
        failed = self.results["failed"]
        
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed} ({passed/total_tests*100:.1f}%)", "INFO")
        self.log(f"Failed: {failed} ({failed/total_tests*100:.1f}%)", "ERROR" if failed > 0 else "INFO")
        
        self.log(f"\n{'Test Name':<30} {'Status':<10} {'Duration':<10}")
        self.log("-" * 50)
        
        for test in self.results["tests"]:
            status = "PASS" if test["passed"] else "FAIL"
            duration = f"{test.get('duration', 0):.2f}s"
            self.log(f"{test['name']:<30} {status:<10} {duration:<10}")
        
        # Save detailed report
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nDetailed report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if failed == 0 else 1)


def main():
    parser = argparse.ArgumentParser(description="GSwarm Comprehensive Test Script")
    parser.add_argument("--host", default="http://localhost:8080", help="GSwarm host URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", help="Run specific test (device, cost, predictor, scheduler, api, e2e)")
    
    args = parser.parse_args()
    
    tester = GSwarmTester(host_url=args.host, verbose=args.verbose)
    
    if args.test:
        # Run specific test
        test_map = {
            "device": tester.test_device_utilities,
            "cost": tester.test_cost_models,
            "predictor": tester.test_predictor,
            "scheduler": tester.test_schedulers,
            "api": tester.test_host_api,
            "e2e": tester.test_end_to_end
        }
        
        if args.test in test_map:
            test_func = test_map[args.test]
            passed = test_func()
            sys.exit(0 if passed else 1)
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {', '.join(test_map.keys())}")
            sys.exit(1)
    else:
        # Run all tests
        tester.run_all_tests()


if __name__ == "__main__":
    main()