"""
Example vLLM Service Implementation

This example shows how to implement a custom model service for vLLM models.
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger
import sys
import os

# Add parent directory to path to import base_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gswarm.model.base_service import ModelService, ServiceConfig, BatchModelService


class VLLMService(BatchModelService):
    """
    vLLM model service implementation.
    
    This service wraps vLLM for serving large language models.
    """
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.engine = None
        self.tokenizer = None
        
    async def start(self) -> None:
        """Initialize vLLM engine"""
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            from transformers import AutoTokenizer
            
            logger.info(f"Loading model from {self.config.model_path}")
            
            # Configure vLLM engine
            engine_args = AsyncEngineArgs(
                model=self.config.model_path,
                dtype="auto",
                max_model_len=self.config.extra_args.get("max_model_len", 2048),
                gpu_memory_utilization=self.config.extra_args.get("gpu_memory_utilization", 0.9),
                device=self.config.device,
            )
            
            # Initialize engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            logger.info("vLLM engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to start vLLM service: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop vLLM engine and clean up resources"""
        if self.engine:
            logger.info("Stopping vLLM engine...")
            # vLLM doesn't have explicit shutdown, just let it be garbage collected
            self.engine = None
            self.tokenizer = None
            logger.info("vLLM engine stopped")
    
    async def inference(self, inputs: Any, parameters: Dict[str, Any]) -> Any:
        """
        Perform text generation inference.
        
        Args:
            inputs: Text prompt or dict with 'prompt' key
            parameters: Generation parameters (temperature, max_tokens, etc.)
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized")
        
        # Extract prompt
        if isinstance(inputs, str):
            prompt = inputs
        elif isinstance(inputs, dict) and "prompt" in inputs:
            prompt = inputs["prompt"]
        else:
            raise ValueError("Invalid input format. Expected string or dict with 'prompt' key")
        
        # Set default parameters
        params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            **parameters
        }
        
        # Generate
        from vllm import SamplingParams
        sampling_params = SamplingParams(**params)
        
        request_id = f"req-{asyncio.get_event_loop().time()}"
        results = []
        
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            results.append(output)
        
        # Return the final output
        if results:
            final_output = results[-1]
            return {
                "text": final_output.outputs[0].text,
                "finish_reason": final_output.outputs[0].finish_reason,
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids)
            }
        
        return {"error": "No output generated"}
    
    async def inference_batch(
        self,
        inputs_list: List[Any],
        parameters_list: List[Dict[str, Any]]
    ) -> List[Any]:
        """Perform batch inference"""
        # For simplicity, process sequentially
        # In production, you'd want to use vLLM's native batch processing
        results = []
        for inputs, params in zip(inputs_list, parameters_list):
            result = await self.inference(inputs, params)
            results.append(result)
        return results
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        status = await super().get_status()
        
        if self.engine:
            # Add vLLM specific status
            status.update({
                "engine": "vllm",
                "model_loaded": True,
                "tokenizer": self.tokenizer.__class__.__name__ if self.tokenizer else None,
            })
        else:
            status.update({
                "engine": "vllm",
                "model_loaded": False,
            })
        
        return status


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Model Service")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace model ID")
    parser.add_argument("--port", type=int, default=8080, help="Service port")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    
    args = parser.parse_args()
    
    config = ServiceConfig(
        name="vllm-service",
        model_path=args.model,
        port=args.port,
        device=args.device,
        max_batch_size=32,
        extra_args={
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
    )
    
    service = VLLMService(config)
    service.run()


if __name__ == "__main__":
    main()