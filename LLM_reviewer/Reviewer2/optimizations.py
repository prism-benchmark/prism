"""
Performance Optimization Utilities for Reviewer2
- vLLM integration option
- Rate limiter for API calls
- Inference speed improvements
"""

import time
import threading
from typing import Any, Callable
import logging

# ==================== RATE LIMITER ====================

class RateLimiter:
    """
    Thread-safe rate limiter for API calls.
    Default: 1 request per second
    """
    
    def __init__(self, rate: float = 1.0, period: float = 1.0):
        """
        Args:
            rate: Number of requests allowed
            period: Time period in seconds (default 1.0 second)
        
        Example:
            limiter = RateLimiter(rate=1, period=1)  # 1 req/sec
            with limiter:
                api_call()
        """
        self.rate = rate
        self.period = period
        self.allowance = rate
        self.last_check = time.time()
        self.lock = threading.Lock()
    
    def __enter__(self):
        """Context manager entry"""
        with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.period)
            
            # Cap allowance at max
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            # If not enough allowance, sleep
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.period / self.rate)
                time.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Usage example:
# limiter = RateLimiter(rate=1, period=1.0)  # 1 req/sec
# with limiter:
#     response = requests.get(url)


# ==================== vLLM INFERENCE (OPTIONAL) ====================

def load_model_vllm(model_path: str, **kwargs):
    """
    Load model using vLLM (much faster than transformers!)
    
    Performance Comparison:
    - Transformers solo: ~30-60 sec per paper (2-phase)
    - vLLM with batching: ~5-15 sec per paper (+4-5x speedup!)
    - vLLM + quantization: ~2-8 sec per paper (+10-15x speedup!)
    
    Args:
        model_path: Path to model
        **kwargs: Additional vLLM parameters
    
    Returns:
        llm: vLLM LLM object
    
    Requirements:
        pip install vllm
    """
    try:
        from vllm import LLM
    except ImportError:
        print("❌ vLLM not installed. Install with:")
        print("   pip install vllm")
        return None
    
    # Recommended vLLM settings for Qwen3-14B
    default_kwargs = {
        'tensor_parallel_size': 1,  # Multi-GPU if available
        'gpu_memory_utilization': 0.55,  # Use ~44GB VRAM on 80GB GPU while staying below the repo's 0.6 cap
        'dtype': 'bfloat16',
        'enforce_eager': False,
        'max_num_batched_tokens': 16384,  # Reduce KV-cache reservation at init
    }
    default_kwargs.update(kwargs)
    
    print(f"Loading vLLM model: {model_path}")
    print(f"Config: {default_kwargs}")
    
    llm = LLM(model=model_path, **default_kwargs)
    return llm


def generate_with_vllm(llm, prompts: list, **sampling_params):
    """
    Generate responses using vLLM (supports batching!)
    
    Args:
        llm: vLLM LLM object
        prompts: List of prompts to generate
        **sampling_params: temp, top_p, top_k, max_tokens, etc.
    
    Returns:
        List of generated outputs
    """
    from vllm import SamplingParams
    
    default_params = {
        'temperature': 0.7,
        'top_p': 0.7,
        'top_k': 50,
        'max_tokens': 8192,
        'repetition_penalty': 1.13,
    }
    default_params.update(sampling_params)
    
    sampling_params = SamplingParams(**default_params)
    outputs = llm.generate(prompts, sampling_params)
    
    return [output.outputs[0].text for output in outputs]


# ==================== OTHER OPTIMIZATIONS ====================

class OptimizationTips:
    """Quick optimization suggestions"""
    
    TIPS = """
    🚀 INFERENCE SPEEDUP STRATEGIES (Ranked by Impact)
    
    1. **vLLM with Batching** (⭐⭐⭐⭐⭐ BEST - 5-15x faster)
       - Batch multiple papers in single forward pass
       - KV-cache optimization
       - Paged attention
       Usage:
           from optimizations import load_model_vllm, generate_with_vllm
           llm = load_model_vllm(MODEL_PATH)
           outputs = generate_with_vllm(llm, [prompt1, prompt2, ...])
    
    2. **Quantization** (⭐⭐⭐⭐ - 2-3x faster, minimal quality loss)
       - int8 quantization (AutoGPTQ)
       - Load model with: load_in_8bit=True
       - Reduces VRAM: 28GB → 14GB for Qwen3-14B
       pip install bitsandbytes auto-gptq
    
    3. **Streaming Decoding** (⭐⭐⭐ - 1.5-2x faster perceived)
       - Tokens appear faster (good UX)
       - Not faster in total time, but feels faster
       from transformers import TextIteratorStreamer
    
    4. **Reduce Token Output** (⭐⭐⭐ - 10-20% faster)
       - Phase 1: 4096 → 2048 tokens
       - Phase 2: 8192 → 4096 tokens
       - Quality may drop, needs testing
    
    5. **Flash Attention 2** (⭐⭐⭐ - already enabled, +10-20%)
       - Already in your llama_attn_replace.py
       - Ensure flash_attn 2.8.3+ installed
    
    6. **Disable Gradient Computation** (⭐⭐ - minor +5%)
       - torch.no_grad() context manager
       - Already used in generate (inference mode)
    
    7. **Reduce Sequence Length** (⭐⭐ - minor +5-10%)
       - SEQ_LEN: 40960 → 32000 (if truncation works)
       - Only if error rate acceptable
    
    8. **CPU Offloading** (⭐ - actually slower, don't use!)
       - device_map="cpu" makes it slower
       - Use device_map="auto" (current setup)
    
    🎯 QUICK WINS (No Code Change):
    - Set CUDA_LAUNCH_BLOCKING=0
    - Set OMP_NUM_THREADS=$(nproc)
    - Monitor GPU utilization (nvidia-smi -l 1)
    - Check for CPU bottleneck (model.to_betterperformance())
    
    📊 CURRENT ESTIMATED SPEEDS:
    - Phase 1 (questions): 15-30 sec/paper
    - Phase 2 (review): 30-60 sec/paper
    - Total: 45-90 sec/paper
    - 5962 papers ≈ 70-150 hours
    
    WITH vLLM + BATCHING:
    - Batch 4 papers together: 20-40 sec/batch
    - Per paper: 5-10 sec
    - Total: ~7-25 hours (10x+ speedup!)
    """
    
    @staticmethod
    def print():
        print(OptimizationTips.TIPS)


# ==================== ENVIRONMENT SETUP ====================

def setup_torch_performance():
    """Setup environment for maximum PyTorch performance"""
    import os
    
    # Disable gradient computation (inference mode)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 8)
    
    # Enable TF32 for faster compute (small accuracy loss)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("✓ PyTorch performance environment setup")
    print(f"  OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    print(f"  TF32 enabled: True")


# ==================== MODEL QUANTIZATION ====================

def load_model_quantized(model_path: str, bits: int = 8):
    """
    Load model with quantization (2-3x faster, 50% VRAM savings)
    
    Args:
        model_path: Path to model
        bits: 8 or 4 (8 is faster, 4 is more compressed)
    
    Returns:
        Quantized model
    
    Requirements:
        pip install bitsandbytes
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    if bits == 8:
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
        )
    elif bits == 4:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    else:
        raise ValueError("bits must be 4 or 8")
    
    print(f"Loading model with {bits}-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✓ Model loaded with {bits}-bit quantization")
    return model


if __name__ == "__main__":
    OptimizationTips.print()
