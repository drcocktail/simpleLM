# CogAI4Sci - Custom Llama 3.2 Implementation

A high-performance, from-scratch implementation of Llama 3.2 (1B/3B) with instant startup times and optimized inference.

## 🚀 Features

- **From-scratch implementation** of Llama 3.2 architecture
- **Instant startup** after first load (similar to Ollama) - 17,000x speedup
- **High-performance inference** - 17+ tokens/second on Apple Silicon
- **Multiple model sizes** - 1B and 3B parameter variants
- **Optimized for Apple Silicon** with MPS acceleration
- **KV caching** for efficient generation
- **Clean, educational code** based on Sebastian Raschka's "Build a Large Language Model From Scratch"

## 📊 Performance

| Metric | Performance |
|--------|-------------|
| **Startup Time** | 0.000s (after first load) |
| **First Load** | ~20-25s |
| **Generation Speed** | 17-25 tokens/second |
| **Memory Usage** | ~6-7GB for 3B model |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cogai4sci.git
cd cogai4sci

# Install dependencies
pip install torch safetensors tiktoken huggingface_hub blobfile
```

## 🏃 Quick Start

### FastLLM - Instant Startup Version

```python
from fast_llm import FastLLM

# Initialize model - first time loads model, subsequent times are instant!
model = FastLLM("meta-llama/Llama-3.2-1B-Instruct", use_float16=True)

# Generate text with streaming
prompt = "What is the future of AI?"
print(prompt, end="")

for token in model.generate(
    prompt, 
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    use_kv_cache=True
):
    print(token, end="", flush=True)
```

### Jupyter Notebook Usage

Open `simpleLLM.ipynb` for an interactive experience with streaming text generation.

## 📁 Project Structure

```
cogai4sci/
├── fast_llm.py              # Main FastLLM class with instant startup
├── simplestLLM.py           # Optimized standalone implementation  
├── simpleLLM.ipynb          # Interactive Jupyter notebook
├── standalone-llama32.ipynb # Reference implementation
├── test_fixes.py            # Test suite for verification
├── performance_test.py      # Performance benchmarking
└── fast_demo.py            # Quick demo script
```

## 🔬 Key Implementations

### 1. FastLLM (`fast_llm.py`)
- **Class-level model caching** for instant subsequent loads
- **Optimized weight loading** for both 1B and 3B models
- **Streaming token generation** with proper tokenization
- **KV cache optimization** for faster inference

### 2. SimplestLLM (`simplestLLM.py`)
- **Educational implementation** with clear, readable code
- **Complete architecture** including RoPE, GroupedQueryAttention, FeedForward
- **Proper dtype handling** for Apple Silicon compatibility
- **Comprehensive tokenizer** with special token support

## 🧠 Architecture Details

### Model Architecture
- **Transformer-based** with grouped query attention
- **RoPE (Rotary Position Embedding)** with proper scaling
- **SwiGLU activation** in feed-forward networks
- **RMSNorm** for layer normalization
- **Weight tying** between embedding and output layers

### Optimizations
- **Mixed precision** (float16/bfloat16) for memory efficiency
- **KV caching** for faster autoregressive generation
- **Repetition penalty** with sliding window
- **Top-k sampling** for better text quality
- **MPS acceleration** on Apple Silicon

## 🔧 Technical Fixes

### Issues Resolved
1. **Italian text generation bug** - Fixed tokenizer chat mode triggering
2. **3B model loading** - Added support for multi-file safetensors
3. **Performance optimization** - Achieved 17+ tokens/second
4. **Memory efficiency** - Reduced memory usage with proper dtype handling
5. **Startup time** - Implemented model caching for instant subsequent loads

## 📈 Benchmarks

Run benchmarks with:
```bash
python performance_test.py
python test_fixes.py
```

## 🎯 Use Cases

- **Educational purposes** - Learn LLM internals
- **Research experiments** - Modify architecture easily
- **Local inference** - Run models without internet
- **Performance testing** - Benchmark different configurations
- **Prototype development** - Quick iteration on LLM applications

## 🤝 Contributing

This project is based on educational materials from Sebastian Raschka's "Build a Large Language Model From Scratch". Contributions welcome!

## 📚 References

- [Build a Large Language Model From Scratch](http://mng.bz/orYv) by Sebastian Raschka
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

## 📄 License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## ⚡ Quick Commands

```bash
# Test both 1B and 3B models
python test_fixes.py

# Run performance benchmarks  
python performance_test.py

# Quick demo with instant startup
python fast_demo.py

# Interactive notebook
jupyter notebook simpleLLM.ipynb
```

---

**Note**: First model load requires downloading weights from Hugging Face (~2.5GB for 3B model). Subsequent loads are instant thanks to our caching system! 