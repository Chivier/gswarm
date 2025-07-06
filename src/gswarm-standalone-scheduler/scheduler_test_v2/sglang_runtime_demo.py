#!/usr/bin/env python3
"""
SGLang Runtime API demo for PD-separated inference
This uses SGLang's Python Runtime API instead of launching separate servers
"""

import time
import asyncio
from typing import Dict, List, Optional
import sys

try:
    import sglang as sgl
    from sglang import Runtime, set_default_backend
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    print("ERROR: SGLang not installed!")
    print("Please install with one of:")
    print("  pip install sglang")
    print("  pip install 'sglang[all]'")
    sys.exit(1)


class SGLangPDDemo:
    """Demo of PD-separated inference using SGLang Runtime API"""
    
    def __init__(self, model_path: str = "microsoft/phi-2"):
        self.model_path = model_path
        self.prefill_runtime = None
        self.decode_runtime = None
        
    def initialize(self):
        """Initialize separate runtimes for prefill and decode"""
        print(f"Initializing SGLang runtimes for {self.model_path}...")
        
        # Prefill runtime - optimized for prompt processing
        print("  Creating prefill runtime...")
        self.prefill_runtime = Runtime(
            model_path=self.model_path,
            tp_size=1,
            max_batch_size=8,
            mem_fraction_static=0.4,  # Less memory for prefill
        )
        
        # Decode runtime - optimized for generation
        print("  Creating decode runtime...")
        self.decode_runtime = Runtime(
            model_path=self.model_path,
            tp_size=1,
            max_batch_size=16,  # Larger batch for decode
            mem_fraction_static=0.5,  # More memory for KV cache
        )
        
        print("✓ Runtimes initialized")
        
    def shutdown(self):
        """Shutdown runtimes"""
        if self.prefill_runtime:
            self.prefill_runtime.shutdown()
        if self.decode_runtime:
            self.decode_runtime.shutdown()
            
    @sgl.function
    def prefill_phase(s, prompt):
        """SGLang function for prefill phase"""
        s += prompt
        # Generate just 1 token to build KV cache
        s += sgl.gen("first_token", max_tokens=1, temperature=0.0)
        
    @sgl.function  
    def decode_phase(s, prompt, max_tokens=50):
        """SGLang function for decode phase"""
        # In real implementation, we'd restore KV cache here
        # For demo, we simulate by providing the prompt
        s += prompt
        s += sgl.gen("output", max_tokens=max_tokens, temperature=0.7, top_p=0.9)
        
    def process_request(self, prompt: str, max_tokens: int = 50) -> Dict:
        """Process a request with PD separation"""
        
        # Phase 1: Prefill
        print(f"\n[Prefill] Processing: {prompt[:40]}...")
        set_default_backend(self.prefill_runtime)
        
        prefill_start = time.time()
        prefill_state = self.prefill_phase.run(prompt=prompt)
        prefill_time = time.time() - prefill_start
        
        # Extract first token (simulating KV cache creation)
        first_token = prefill_state["first_token"]
        print(f"[Prefill] Complete in {prefill_time:.3f}s")
        
        # Phase 2: Decode  
        print(f"[Decode] Generating {max_tokens} tokens...")
        set_default_backend(self.decode_runtime)
        
        decode_start = time.time()
        # In real implementation, we'd pass KV cache
        # For demo, we pass prompt + first token
        decode_state = self.decode_phase.run(
            prompt=prompt + first_token,
            max_tokens=max_tokens
        )
        decode_time = time.time() - decode_start
        
        generated_text = decode_state["output"]
        print(f"[Decode] Complete in {decode_time:.3f}s")
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "first_token": first_token
        }
        
    def run_benchmark(self, prompts: List[str]):
        """Run benchmark on multiple prompts"""
        print("\n" + "="*70)
        print("RUNNING PD-SEPARATED INFERENCE BENCHMARK")
        print("="*70)
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n--- Request {i+1}/{len(prompts)} ---")
            result = self.process_request(prompt, max_tokens=50)
            results.append(result)
            
            print(f"Generated: {result['generated_text'][:80]}...")
            
        # Summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        avg_prefill = sum(r["prefill_time"] for r in results) / len(results)
        avg_decode = sum(r["decode_time"] for r in results) / len(results)
        avg_total = sum(r["total_time"] for r in results) / len(results)
        
        print(f"\nAverage times for {len(results)} requests:")
        print(f"  Prefill: {avg_prefill:.3f}s")
        print(f"  Decode:  {avg_decode:.3f}s")
        print(f"  Total:   {avg_total:.3f}s")
        
        print(f"\nTime distribution:")
        print(f"  Prefill: {(avg_prefill/avg_total)*100:.1f}%")
        print(f"  Decode:  {(avg_decode/avg_total)*100:.1f}%")
        
        return results


def run_simple_sglang_example():
    """Run a simple SGLang example without PD separation"""
    print("\n" + "="*70)
    print("SIMPLE SGLANG EXAMPLE (No PD Separation)")
    print("="*70)
    
    # Create runtime
    print("Creating SGLang runtime...")
    runtime = Runtime(
        model_path="microsoft/phi-2",
        tp_size=1,
        max_batch_size=8
    )
    set_default_backend(runtime)
    
    @sgl.function
    def simple_generation(s, prompt, max_tokens=50):
        s += prompt
        s += sgl.gen("output", max_tokens=max_tokens, temperature=0.7)
    
    # Test generation
    prompt = "What is machine learning?"
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    
    start_time = time.time()
    state = simple_generation.run(prompt=prompt)
    total_time = time.time() - start_time
    
    print(f"Generated: {state['output']}")
    print(f"Time: {total_time:.3f}s")
    
    # Shutdown
    runtime.shutdown()
    print("\n✓ Runtime shutdown")


def main():
    """Main demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SGLang Runtime API demo")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", 
                      help="Model to use (default: microsoft/phi-2)")
    parser.add_argument("--simple", action="store_true", 
                      help="Run simple example without PD separation")
    
    args = parser.parse_args()
    
    if args.simple:
        run_simple_sglang_example()
    else:
        # Run PD-separated demo
        demo = SGLangPDDemo(model_path=args.model)
        
        try:
            demo.initialize()
            
            # Test prompts
            test_prompts = [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms.",
                "How do neural networks work?",
                "What are the benefits of deep learning?",
                "Describe the transformer architecture.",
            ]
            
            demo.run_benchmark(test_prompts)
            
        finally:
            demo.shutdown()
            print("\n✓ All runtimes shutdown")


if __name__ == "__main__":
    main()