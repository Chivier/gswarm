#!/usr/bin/env python3
"""
Unit tests for device utilities
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from gswarm.utils import parse_device, format_device, normalize_device, is_same_device, DeviceInfo


class TestDeviceUtils(unittest.TestCase):
    """Test device parsing and formatting utilities"""
    
    def test_parse_new_format(self):
        """Test parsing new device format"""
        # Standard format
        device = parse_device("node1:cuda:0")
        self.assertEqual(device.client, "node1")
        self.assertEqual(device.device_type, "cuda")
        self.assertEqual(device.device_id, 0)
        self.assertEqual(device.full_name, "node1:cuda:0")
        
        # Different client names
        device = parse_device("gpu-server-west:cuda:3")
        self.assertEqual(device.client, "gpu-server-west")
        self.assertEqual(device.device_id, 3)
        
        # CPU device
        device = parse_device("worker1:cpu:0")
        self.assertEqual(device.device_type, "cpu")
    
    def test_parse_legacy_format(self):
        """Test parsing legacy device format"""
        # Legacy CUDA format
        device = parse_device("cuda:0")
        self.assertEqual(device.client, "localhost")
        self.assertEqual(device.device_type, "cuda")
        self.assertEqual(device.device_id, 0)
        
        # Legacy with different ID
        device = parse_device("cuda:7")
        self.assertEqual(device.device_id, 7)
        
        # With default client
        device = parse_device("cuda:2", default_client="mynode")
        self.assertEqual(device.client, "mynode")
    
    def test_format_device(self):
        """Test device formatting"""
        # Standard formatting
        formatted = format_device("node1", "cuda", 0)
        self.assertEqual(formatted, "node1:cuda:0")
        
        # Different values
        formatted = format_device("gpu-cluster", "cuda", 3)
        self.assertEqual(formatted, "gpu-cluster:cuda:3")
        
        # CPU device
        formatted = format_device("worker", "cpu", 0)
        self.assertEqual(formatted, "worker:cpu:0")
    
    def test_normalize_device(self):
        """Test device normalization"""
        # Legacy to normalized
        normalized = normalize_device("cuda:0")
        self.assertEqual(normalized, "localhost:cuda:0")
        
        # Already normalized
        normalized = normalize_device("node1:cuda:0")
        self.assertEqual(normalized, "node1:cuda:0")
        
        # With default client
        normalized = normalize_device("cuda:1", default_client="cluster")
        self.assertEqual(normalized, "cluster:cuda:1")
    
    def test_is_same_device(self):
        """Test device comparison"""
        # Same devices
        self.assertTrue(is_same_device("node1:cuda:0", "node1:cuda:0"))
        
        # Legacy vs normalized
        self.assertTrue(is_same_device("cuda:0", "localhost:cuda:0"))
        
        # Different devices
        self.assertFalse(is_same_device("node1:cuda:0", "node2:cuda:0"))
        self.assertFalse(is_same_device("node1:cuda:0", "node1:cuda:1"))
        
        # With default client
        self.assertTrue(is_same_device("cuda:0", "worker:cuda:0", default_client="worker"))
    
    def test_device_info_properties(self):
        """Test DeviceInfo dataclass properties"""
        device = DeviceInfo(client="node1", device_type="cuda", device_id=0)
        self.assertEqual(device.full_name, "node1:cuda:0")
        self.assertEqual(device.legacy_name, "cuda:0")
        
        # Test string representation
        self.assertEqual(str(device), "node1:cuda:0")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Invalid format should raise ValueError
        with self.assertRaises(ValueError):
            parse_device("invalid_format")
        
        with self.assertRaises(ValueError):
            parse_device("node1:cuda")
        
        with self.assertRaises(ValueError):
            parse_device("node1:cuda:abc")  # Non-numeric ID
        
        # Empty or None
        with self.assertRaises(ValueError):
            parse_device("")
    
    def test_special_characters(self):
        """Test handling of special characters in client names"""
        # Hyphens and underscores
        device = parse_device("gpu_server-01:cuda:0")
        self.assertEqual(device.client, "gpu_server-01")
        
        # Numbers in client name
        device = parse_device("node123:cuda:0")
        self.assertEqual(device.client, "node123")


if __name__ == "__main__":
    unittest.main()