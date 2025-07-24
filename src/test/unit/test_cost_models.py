#!/usr/bin/env python3
"""
Unit tests for cost models
"""

import unittest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from gswarm.model import CostModel, LLMCostModel, SDCostModel, get_estimation_cost, update_predictor


class TestCostModels(unittest.TestCase):
    """Test cost prediction models"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for model storage
        self.temp_dir = tempfile.mkdtemp()
        os.environ['GSWARM_MODEL_DIR'] = self.temp_dir
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_llm_cost_model(self):
        """Test LLM cost model"""
        model = LLMCostModel()
        
        # Test prediction by prompt length
        tokens = model.predict(prompt_length=100)
        self.assertGreater(tokens, 0)
        
        # Test prediction by string
        tokens = model.predict_str("Hello, how are you?")
        self.assertGreater(tokens, 0)
        
        # Test with longer prompt
        long_prompt = "Explain quantum computing " * 20
        tokens_long = model.predict_str(long_prompt)
        tokens_short = model.predict_str("Short")
        self.assertGreater(tokens_long, tokens_short)
    
    def test_sd_cost_model(self):
        """Test Stable Diffusion cost model"""
        model = SDCostModel()
        
        # Test basic prediction
        time = model.predict("stable-diffusion", "node1:cuda:0", 512, 512)
        self.assertGreater(time, 0)
        
        # Test larger image takes more time
        time_small = model.predict("sd", "cuda:0", 256, 256)
        time_large = model.predict("sd", "cuda:0", 1024, 1024)
        self.assertGreater(time_large, time_small)
        
        # Test device normalization
        time1 = model.predict("sd", "cuda:0", 512, 512)
        time2 = model.predict("sd", "localhost:cuda:0", 512, 512)
        self.assertEqual(time1, time2)
    
    def test_unified_cost_model(self):
        """Test unified CostModel interface"""
        model = CostModel()
        
        # Test LLM prediction
        llm_time = model.predict("gpt-4", "node1:cuda:0", {"prompt": "Hello world"})
        self.assertGreater(llm_time, 0)
        
        # Test SD prediction
        sd_time = model.predict("stable-diffusion", "node2:cuda:1", {"height": 512, "width": 512})
        self.assertGreater(sd_time, 0)
        
        # Test auto-detection
        llama_time = model.predict("llama-2-7b", "cuda:0", {"prompt": "Test"})
        self.assertGreater(llama_time, 0)
        
        sdxl_time = model.predict("stable-diffusion-xl", "cuda:1", {"height": 1024, "width": 1024})
        self.assertGreater(sdxl_time, 0)
    
    def test_model_type_detection(self):
        """Test automatic model type detection"""
        model = CostModel()
        
        # LLM models
        llm_models = ["gpt-4", "gpt-3.5-turbo", "llama-2-7b", "mistral-7b", "claude-2"]
        for model_name in llm_models:
            time = model.predict(model_name, "cuda:0", {"prompt": "Test"})
            self.assertGreater(time, 0)
        
        # SD models
        sd_models = ["stable-diffusion", "stable-diffusion-xl", "sdxl", "dalle-3"]
        for model_name in sd_models:
            time = model.predict(model_name, "cuda:0", {"height": 512, "width": 512})
            self.assertGreater(time, 0)
    
    def test_model_update(self):
        """Test model update functionality"""
        model = CostModel()
        
        # Update LLM model
        model.update("gpt-4", "node1:cuda:0", 
                    {"prompt": "Hi", "output": "Hello there!"}, 
                    actual_time=0.5)
        
        # Update SD model
        model.update("stable-diffusion", "node2:cuda:1",
                    {"height": 512, "width": 512, "processing_time": 2.5},
                    actual_time=2.5)
        
        # No errors should occur
        self.assertTrue(True)
    
    def test_legacy_api(self):
        """Test backward compatibility with legacy API"""
        # Test LLM
        llm_cost = get_estimation_cost(
            model_type="llm",
            model_name="gpt-4",
            device="cuda:0",
            data_features=[{"prompt": "Hello, world!"}]
        )
        self.assertGreater(llm_cost, 0)
        
        # Test SD
        sd_cost = get_estimation_cost(
            model_type="diffusion",
            model_name="stable-diffusion-v1-5",
            device="cuda:0",
            data_features=[{"height": 512, "width": 512}]
        )
        self.assertGreater(sd_cost, 0)
        
        # Test update
        update_predictor(
            model_type="llm",
            model_name="gpt-4",
            device="cuda:0",
            data_features=[{"prompt": "Test", "output": "Response"}]
        )
    
    def test_device_specific_models(self):
        """Test that models are device-specific"""
        model = CostModel()
        
        # Different devices should potentially give different predictions
        # (though in practice they might be the same initially)
        time1 = model.predict("gpt-4", "node1:cuda:0", {"prompt": "Test"})
        time2 = model.predict("gpt-4", "node2:cuda:1", {"prompt": "Test"})
        
        # Both should be valid predictions
        self.assertGreater(time1, 0)
        self.assertGreater(time2, 0)
    
    def test_error_handling(self):
        """Test error handling in cost models"""
        model = CostModel()
        
        # Missing required inputs for LLM
        with self.assertRaises(KeyError):
            model.predict("gpt-4", "cuda:0", {})
        
        # Missing required inputs for SD
        with self.assertRaises(KeyError):
            model.predict("stable-diffusion", "cuda:0", {"height": 512})
        
        # Invalid model type with explicit type
        with self.assertRaises(ValueError):
            model.predict("unknown-model", "cuda:0", {"prompt": "Test"}, model_type="unknown")


if __name__ == "__main__":
    unittest.main()