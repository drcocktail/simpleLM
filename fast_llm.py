import torch
import torch.nn as nn
from transformers import AutoConfig
import os
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import time
import pickle
import hashlib

# Import the model architecture from simplestLLM
from simplestLLM import (
    FeedForward, precompute_rope_params, compute_rope, SharedBuffers,
    GroupedQueryAttention, TransformerBlock, Llama3Model, Tokenizer, rescale_theta
)

class FastLLM:
    """
    Optimized LLM with fast startup times - similar to Ollama's approach
    """
    _cached_models = {}  # Class-level cache for loaded models
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", context_length=4096, use_float16=True, force_reload=False):
        self.model_name = model_name
        self.context_length = context_length
        self.use_float16 = use_float16
        
        # Create a cache key based on model configuration
        cache_key = f"{model_name}_{context_length}_{use_float16}"
        
        # Check if model is already loaded in memory
        if cache_key in FastLLM._cached_models and not force_reload:
            print("⚡ Using cached model - instant startup!")
            cached_data = FastLLM._cached_models[cache_key]
            self.model = cached_data['model']
            self.tokenizer = cached_data['tokenizer']
            self.config = cached_data['config']
            self.device = cached_data['device']
            self.dtype = cached_data['dtype']
            return
        
        print("🔄 Loading model for the first time (this will be cached)...")
        start_time = time.time()
        
        # Set up device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS for acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        self.dtype = torch.float16 if use_float16 and self.device.type != 'cpu' else torch.float32
        
        if use_float16 and self.device.type != 'cpu':
            print("Using float16 for memory efficiency and speed")
        else:
            print("Using float32 for maximum precision")
        
        # Load model configuration and weights
        self._load_model_and_tokenizer()
        
        # Cache the loaded model
        FastLLM._cached_models[cache_key] = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'config': self.config,
            'device': self.device,
            'dtype': self.dtype
        }
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded and cached in {load_time:.2f}s (next startup will be instant!)")
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer (only called on first load)"""
        # Get config from Hugging Face
        hf_config = AutoConfig.from_pretrained(self.model_name).to_dict()
        
        # Rescale RoPE theta based on context length
        rope_base = rescale_theta(
            theta_old=hf_config["rope_theta"],
            context_length_old=hf_config["max_position_embeddings"],
            context_length_new=self.context_length
        )
        
        # Create model config
        self.config = {
            "vocab_size": hf_config["vocab_size"],
            "context_length": self.context_length,
            "emb_dim": hf_config["hidden_size"],
            "n_heads": hf_config["num_attention_heads"],
            "n_layers": hf_config["num_hidden_layers"],
            "hidden_dim": hf_config["intermediate_size"],
            "n_kv_groups": hf_config["num_key_value_heads"],
            "rope_base": rope_base,
            "dtype": self.dtype,
            "rope_freq": {
                 "factor": hf_config["rope_scaling"]["factor"],
                 "low_freq_factor": hf_config["rope_scaling"]["low_freq_factor"],
                 "high_freq_factor": hf_config["rope_scaling"]["high_freq_factor"],
                 "original_context_length": hf_config["rope_scaling"]["original_max_position_embeddings"],
            }
        }
        
        # Create model
        self.model = Llama3Model(self.config).to(self.device)
        
        # Load weights efficiently
        self._load_weights()
        
        # Setup tokenizer
        self._setup_tokenizer()
        
        self.model.eval()
    
    def _load_weights(self):
        """Efficiently load model weights"""
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        
        # Determine model size from embedding dimension
        model_size = "1B" if self.config["emb_dim"] == 2048 else "3B"
        
        if model_size == "1B":
            # 1B model has a single safetensors file
            weights_file = hf_hub_download(
                repo_id=self.model_name,
                filename="model.safetensors",
                local_dir=f"./{self.model_name.replace('/', '_')}"
            )
            combined_weights = load_file(weights_file)
        else:
            # 3B model has multiple safetensors files
            combined_weights = {}
            for i in range(1, 3):  # Files are numbered 1 and 2
                weights_file = hf_hub_download(
                    repo_id=self.model_name,
                    filename=f"model-0000{i}-of-00002.safetensors",
                    local_dir=f"./{self.model_name.replace('/', '_')}"
                )
                current_weights = load_file(weights_file)
                combined_weights.update(current_weights)
        
        # Apply weights to model
        self._load_weights_into_llama(self.model, self.config, combined_weights)
        self.model.to(self.device)
        del combined_weights  # Free memory
    
    def _setup_tokenizer(self):
        """Setup tokenizer"""
        from huggingface_hub import hf_hub_download
        
        tokenizer_file_path = hf_hub_download(
            repo_id=self.model_name, 
            filename="original/tokenizer.model",
            local_dir=f"./{self.model_name.replace('/', '_')}"
        )
        self.tokenizer = Tokenizer(tokenizer_file_path)
    
    def _assign(self, left, right, tensor_name="unknown"):
        """Helper method for weight assignment"""
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        
        if isinstance(right, torch.Tensor):
            return torch.nn.Parameter(right.clone().detach())
        else:
            return torch.nn.Parameter(torch.tensor(right))
    
    def _load_weights_into_llama(self, model, param_config, params):
        """Load weights into the model"""
        model.tok_emb.weight = self._assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

        for l in range(param_config["n_layers"]):
            # Load attention weights
            model.trf_blocks[l].att.W_query.weight = self._assign(
                model.trf_blocks[l].att.W_query.weight,
                params[f"model.layers.{l}.self_attn.q_proj.weight"],
                f"model.layers.{l}.self_attn.q_proj.weight"
            )
            model.trf_blocks[l].att.W_key.weight = self._assign(
                model.trf_blocks[l].att.W_key.weight,
                params[f"model.layers.{l}.self_attn.k_proj.weight"],
                f"model.layers.{l}.self_attn.k_proj.weight"
            )
            model.trf_blocks[l].att.W_value.weight = self._assign(
                model.trf_blocks[l].att.W_value.weight,
                params[f"model.layers.{l}.self_attn.v_proj.weight"],
                f"model.layers.{l}.self_attn.v_proj.weight"
            )
            model.trf_blocks[l].att.out_proj.weight = self._assign(
                model.trf_blocks[l].att.out_proj.weight,
                params[f"model.layers.{l}.self_attn.o_proj.weight"],
                f"model.layers.{l}.self_attn.o_proj.weight"
            )
            model.trf_blocks[l].norm1.weight = self._assign(
                model.trf_blocks[l].norm1.weight,
                params[f"model.layers.{l}.input_layernorm.weight"],
                f"model.layers.{l}.input_layernorm.weight"
            )

            # Load FeedForward weights
            model.trf_blocks[l].ff.fc1.weight = self._assign(
                model.trf_blocks[l].ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            model.trf_blocks[l].ff.fc2.weight = self._assign(
                model.trf_blocks[l].ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            model.trf_blocks[l].ff.fc3.weight = self._assign(
                model.trf_blocks[l].ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )
            model.trf_blocks[l].norm2.weight = self._assign(
                model.trf_blocks[l].norm2.weight,
                params[f"model.layers.{l}.post_attention_layernorm.weight"],
                f"model.layers.{l}.post_attention_layernorm.weight"
            )

        # Load output layer weights
        model.final_norm.weight = self._assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

        if "lm_head.weight" in params.keys():
            model.out_head.weight = self._assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
        else:
            model.out_head.weight = self._assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    
    def generate(self, prompt, max_new_tokens=50, temperature=0.0, top_k=None, repetition_penalty=1.1, use_kv_cache=True):
        """Generate text with the cached model - instant startup!"""
        # Use simple text completion without chat tokens
        token_ids = self.tokenizer.encode(prompt, bos=False, eos=False)
        idx = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Clear any existing cache
        if use_kv_cache:
            self.model.clear_cache()
        
        # First forward pass with full prompt - this is the prefill phase
        with torch.no_grad():
            logits = self.model(idx, use_kv_cache=use_kv_cache, cache_position=None)
        
        # Get logits for the last token and convert to appropriate dtype for sampling
        if self.use_float16:
            logits = logits[:, -1, :].float()  # Convert to float32 for numerical stability in sampling
        else:
            logits = logits[:, -1, :]
        
        # Pre-compute repetition penalty tokens window
        penalty_window = 64
        
        # Generate tokens one by one using KV cache
        for step in range(max_new_tokens):
            # Apply repetition penalty more efficiently
            if repetition_penalty != 1.0 and idx.shape[1] > 1:
                # Only get the last penalty_window tokens
                start_idx = max(0, idx.shape[1] - penalty_window)
                penalty_tokens = idx[0, start_idx:].tolist()
                unique_tokens = set(penalty_tokens)
                
                # Apply penalty in a vectorized way
                for token_id in unique_tokens:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample next token
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Check for special tokens that should stop generation
            token_id = idx_next.item()
            if token_id in [
                self.tokenizer.special_tokens.get("<|end_of_text|>", -1),
                self.tokenizer.special_tokens.get("<|eot_id|>", -1),
                self.tokenizer.special_tokens.get("<|start_header_id|>", -1),
                self.tokenizer.special_tokens.get("<|end_header_id|>", -1)
            ]:
                break
            
            # Decode and yield the token
            token_text = self.tokenizer.decode([idx_next.item()])
            yield token_text
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Check if we've exceeded context length
            if idx.shape[1] >= self.config['context_length']:
                break
            
            # Forward pass for next token using KV cache - this is the decode phase
            if step < max_new_tokens - 1:  # Don't compute on last iteration
                with torch.no_grad():
                    # Pass the single new token and correct cache position
                    logits = self.model(idx_next, use_kv_cache=use_kv_cache, 
                                      cache_position=idx.shape[1]-1)
                
                # Convert to appropriate dtype for sampling
                if self.use_float16:
                    logits = logits[:, -1, :].float()
                else:
                    logits = logits[:, -1, :]
        
        # Clear cache after generation to free memory
        if use_kv_cache:
            self.model.clear_cache()
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models to free memory"""
        cls._cached_models.clear()
        print("🗑️  Model cache cleared")
    
    @classmethod
    def list_cached_models(cls):
        """List all cached models"""
        if not cls._cached_models:
            print("No models currently cached")
        else:
            print("Cached models:")
            for key in cls._cached_models.keys():
                print(f"  - {key}")

# Convenience function for quick usage
def quick_generate(prompt, max_tokens=100, temperature=0.8):
    """Quick generation function - model stays loaded between calls"""
    model = FastLLM()
    
    print(prompt, end="", flush=True)
    for token in model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature):
        print(token, end="", flush=True)
    print()  # New line at end 