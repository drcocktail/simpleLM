import torch
import torch.nn as nn
from transformers import  AutoConfig
import os
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe


# =================================================================================
# Core Model Architecture Code from standalone-llama32.ipynb
# =================================================================================

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Use dtype from config for consistency
        dtype = cfg.get("dtype", torch.float32)
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=dtype, bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=dtype, bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]

class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,
            rope_base=10_000,
            rope_config=None,
            dtype=None
        ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # Fetch buffers using SharedBuffers
        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        self.register_buffer("mask", mask)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        
        # KV Cache buffers - initialized as None, created on first use
        self.kv_cache_k = None
        self.kv_cache_v = None

    def forward(self, x, use_kv_cache=False, cache_position=None):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_kv_groups, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_kv_groups, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)

        # Apply RoPE
        if use_kv_cache and cache_position is not None:
            # For cached generation, apply RoPE with the correct position
            keys = compute_rope(keys, self.cos[cache_position:cache_position+num_tokens], 
                              self.sin[cache_position:cache_position+num_tokens])
            queries = compute_rope(queries, self.cos[cache_position:cache_position+num_tokens], 
                                 self.sin[cache_position:cache_position+num_tokens])
        else:
            keys = compute_rope(keys, self.cos, self.sin)
            queries = compute_rope(queries, self.cos, self.sin)

        # KV Caching logic
        if use_kv_cache:
            if self.kv_cache_k is None or self.kv_cache_v is None:
                # Initialize cache
                self.kv_cache_k = torch.zeros(b, self.num_kv_groups, self.context_length, self.head_dim, 
                                            dtype=keys.dtype, device=keys.device)
                self.kv_cache_v = torch.zeros(b, self.num_kv_groups, self.context_length, self.head_dim, 
                                            dtype=values.dtype, device=values.device)
            
            if cache_position is not None:
                # Update cache with new keys and values
                self.kv_cache_k[:, :, cache_position:cache_position+num_tokens] = keys
                self.kv_cache_v[:, :, cache_position:cache_position+num_tokens] = values
                
                # Use cached keys and values up to current position
                keys = self.kv_cache_k[:, :, :cache_position+num_tokens]
                values = self.kv_cache_v[:, :, :cache_position+num_tokens]
            else:
                # First forward pass - store in cache
                self.kv_cache_k[:, :, :num_tokens] = keys
                self.kv_cache_v[:, :, :num_tokens] = values

        # Expand keys and values to match the number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, seq_len, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, seq_len, head_dim)

        # Compute scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Apply causal mask
        seq_len = keys.shape[2]
        if use_kv_cache and cache_position is not None:
            # For cached generation, only mask future positions
            mask_bool = self.mask.bool()[cache_position:cache_position+num_tokens, :seq_len]
        else:
            mask_bool = self.mask.bool()[:num_tokens, :seq_len]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)

        # Apply attention to values
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
    
    def clear_cache(self):
        """Clear KV cache to free memory"""
        self.kv_cache_k = None
        self.kv_cache_v = None

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
            rope_config=cfg["rope_freq"],
            dtype=torch.float32
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, x, use_kv_cache=False, cache_position=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_kv_cache=use_kv_cache, cache_position=cache_position)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)    # Keep in float32 for stability on MPS
        x = x + shortcut  # Add the original input back

        return x
    
    def clear_cache(self):
        """Clear KV cache for this transformer block"""
        self.att.clear_cache()

class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=torch.float32)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=torch.float32)

    def forward(self, in_idx, use_kv_cache=False, cache_position=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        
        # Pass through transformer blocks with KV cache support
        for block in self.trf_blocks:
            x = block(x, use_kv_cache=use_kv_cache, cache_position=cache_position)
            
        x = self.final_norm(x)
        logits = self.out_head(x)  # Keep in float32 for stability
        return logits
    
    def clear_cache(self):
        """Clear all KV caches"""
        for block in self.trf_blocks:
            block.clear_cache()

# =================================================================================
# Tokenizer Code from standalone-llama32.ipynb
# =================================================================================

class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)
        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        # Add reserved tokens from the notebook
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })
        self.model = tiktoken.Encoding(name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

    def encode(self, text, bos=True, eos=False):
        tokens = [self.special_tokens["<|begin_of_text|>"]] if bos else []
        tokens += self.model.encode(text)
        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)

# =================================================================================
# Main Wrapper Class
# =================================================================================

def rescale_theta(theta_old, context_length_old, context_length_new):
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new

class SimplestLLM:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", context_length=4096, use_float16=True):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS for acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        self.use_float16 = use_float16 and self.device.type != 'cpu'
        self.dtype = torch.float16 if self.use_float16 else torch.float32
        
        if self.use_float16:
            print("Using float16 for memory efficiency and speed")
        else:
            print("Using float32 for maximum precision")
        
        # Get config from Hugging Face
        hf_config = AutoConfig.from_pretrained(model_name).to_dict()

        # Allow configurable context length
        print(f"Setting context length to {context_length} tokens")
        
        # Correctly rescale RoPE theta based on the new context length
        rope_base = rescale_theta(
            theta_old=hf_config["rope_theta"],
            context_length_old=hf_config["max_position_embeddings"],
            context_length_new=context_length
        )

        # Create our local config with proper dtype
        target_dtype = torch.float16 if use_float16 else torch.float32
        
        self.config = {
            "vocab_size": hf_config["vocab_size"],
            "context_length": context_length,
            "emb_dim": hf_config["hidden_size"],
            "n_heads": hf_config["num_attention_heads"],
            "n_layers": hf_config["num_hidden_layers"],
            "hidden_dim": hf_config["intermediate_size"],
            "n_kv_groups": hf_config["num_key_value_heads"],
            "rope_base": rope_base,
            "dtype": target_dtype,  # Add dtype to config
            "rope_freq": {
                 "factor": hf_config["rope_scaling"]["factor"],
                 "low_freq_factor": hf_config["rope_scaling"]["low_freq_factor"],
                 "high_freq_factor": hf_config["rope_scaling"]["high_freq_factor"],
                 "original_context_length": hf_config["rope_scaling"]["original_max_position_embeddings"],
            }
        }
        
        # Instantiate the model with consistent dtype
        self.model = Llama3Model(self.config).to(self.device)
        
        # No need for additional .half() call since dtype is already set in config
        
        # Load weights using the same method as the notebook
        print("Loading weights...")
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        
        # Download the weights file
        weights_file = hf_hub_download(
            repo_id=model_name,
            filename="model.safetensors",
            local_dir=f"./{model_name.replace('/', '_')}"
        )
        combined_weights = load_file(weights_file)
        
        # Load weights using the custom mapping function
        self._load_weights_into_llama(self.model, self.config, combined_weights)
        self.model.to(self.device)
        del combined_weights  # free up memory
        
        self.model.eval() # Set to evaluation mode
        print("Model and weights loaded successfully.")
        
        # Setup tokenizer
        tokenizer_file_path = hf_hub_download(repo_id=model_name, filename="original/tokenizer.model",
                                             local_dir=f"./{model_name.replace('/', '_')}")
        self.tokenizer = Tokenizer(tokenizer_file_path)

    def _assign(self, left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        if isinstance(right, torch.Tensor):
            return torch.nn.Parameter(right.clone().detach())
        else:
            return torch.nn.Parameter(torch.tensor(right))

    def _load_weights_into_llama(self, model, param_config, params):
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
            print("Model uses weight tying.")

    def generate(self, prompt, max_new_tokens=50, temperature=0.0, top_k=None, repetition_penalty=1.1, use_kv_cache=True):
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
                print(f"\nReached maximum context length of {self.config['context_length']} tokens")
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