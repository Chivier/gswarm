#!/bin/bash

# Example curl commands for workflow requests

curl -X POST http://localhost:8080/workflow \
  -H "Content-Type: application/json" \
  -d '{
  "request_id": "req_0000",
  "timestamp": "2025-06-20T12:47:31.136000",
  "workflow_id": "workflow2",
  "input_data": {
    "user_prompt": "Sample prompt for request 0"
  },
  "node_configs": {}
}'

curl -X POST http://localhost:8080/workflow \
  -H "Content-Type: application/json" \
  -d '{
  "request_id": "req_0001",
  "timestamp": "2025-06-20T12:47:34.776360",
  "workflow_id": "workflow1",
  "input_data": {
    "user_prompt": "Sample prompt for request 1"
  },
  "node_configs": {
    "node2": {
      "width": 1024,
      "height": 512
    }
  }
}'

curl -X POST http://localhost:8080/workflow \
  -H "Content-Type: application/json" \
  -d '{
  "request_id": "req_0002",
  "timestamp": "2025-06-20T12:47:38.918338",
  "workflow_id": "workflow1",
  "input_data": {
    "user_prompt": "Sample prompt for request 2"
  },
  "node_configs": {
    "node2": {
      "width": 768,
      "height": 768
    }
  }
}'

