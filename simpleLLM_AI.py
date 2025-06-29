import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
import json

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

def precompute_rope_freqs(
    dim: int, 
    end: int, 
    theta: float = 10000.0,
    freq_config: Optional[Dict] = None
) -> torch.Tensor:
    """
    Precompute the frequency tensor for RoPE.
    """
    # Create position indices
    positions = torch.arange(end, dtype=torch.float)
    
    # Create frequency tensor
    dim_tensor = torch.arange(0, dim, 2, dtype=torch.float)
    
    # Apply rope frequency scaling if provided
    if freq_config is not None:
        # Extract configuration
        factor = freq_config.get("factor", 1.0)
        low_freq_factor = freq_config.get("low_freq_factor", 1.0)
        high_freq_factor = freq_config.get("high_freq_factor", 1.0)
        original_context_length = freq_config.get("original_context_length", end)
        
        # Calculate wavelengths
        inv_freq = 1.0 / (theta ** (dim_tensor / dim))
        wavelengths = 2 * torch.pi / inv_freq
        
        # Apply scaling based on wavelength
        low_freq_wavelen = original_context_length / low_freq_factor
        high_freq_wavelen = original_context_length / high_freq_factor
        
        # Scale low frequencies
        is_low_freq = wavelengths > low_freq_wavelen
        inv_freq_scaled = torch.where(is_low_freq, inv_freq / factor, inv_freq)
        
        # Scale medium frequencies with smooth transition
        smooth_factor = (original_context_length / wavelengths - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        )
        
        # Apply medium frequency scaling
        is_medium_freq = (wavelengths <= low_freq_wavelen) & (wavelengths >= high_freq_wavelen)
        inv_freq_scaled = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_scaled)
        
        # Apply scaled frequencies
        freqs = positions[:, None] * inv_freq_scaled[None, :]
    else:
        # Default RoPE frequencies
        freqs = positions[:, None] * (1.0 / (theta ** (dim_tensor / dim)))[None, :]
    
    # Return precomputed cos and sin values
    return torch.cat([freqs, freqs], dim=-1)

def apply_rotary_emb(
    x: torch.Tensor, 
    freqs: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequencies.
    """
    # x: (batch, heads, seq_len, head_dim)
    seq_len = x.shape[2]
    # Subselect freqs to match the sequence length
    freqs = freqs[:seq_len].to(device)  # Move freqs to the same device as x
    
    # Make sure we're using only half the dimensions for cos/sin
    head_dim = x.shape[3]
    half_head_dim = head_dim // 2
    
    # Calculate cos and sin on the half dimension
    cos_ = freqs[:, :half_head_dim].cos()
    sin_ = freqs[:, :half_head_dim].sin()
    
    # x_even and x_odd now are of shape (batch, heads, seq_len, head_dim//2)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    # Unsqueeze cos_ and sin_ for proper broadcasting
    cos_ = cos_.unsqueeze(0).unsqueeze(0)  # shape: (1,1,seq_len, head_dim//2)
    sin_ = sin_.unsqueeze(0).unsqueeze(0)  # FIX: Use sin_ here instead of cos_
    
    # Apply rotations
    x_rotated = torch.stack(
        [x_even * cos_ - x_odd * sin_,
         x_odd * cos_ + x_even * sin_],
        dim=-1
    ).flatten(-2)
    
    return x_rotated

class SelfAttention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        num_kv_heads: int,
        rope_config: Dict,
        max_seq_length: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        
        # Group size = num query heads per kv head
        self.n_rep = self.num_heads // self.num_kv_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # Rope frequency configuration
        theta = rope_config.get("rope_base", 10000.0)
        self.register_buffer(
            "freqs",
            precompute_rope_freqs(
                self.head_dim, 
                max_seq_length, 
                theta=theta,
                freq_config=rope_config.get("rope_freq")
            )
        )
        
        # Scale factor for attention
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length, hidden_dim = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Get position-specific frequencies for rotary embeddings
        position_freqs = self.freqs[start_pos:start_pos + seq_length]
        
        # Apply rotary embeddings with position-specific frequencies
        q = apply_rotary_emb(q, position_freqs, x.device)
        k = apply_rotary_emb(k, position_freqs, x.device)
        
        # Use key-value cache if provided (optimization)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # Debug prints to see tensor shapes
            print(f"past_k shape: {past_k.shape}, k shape: {k.shape}")
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Save current k, v in the cache
        new_kv_cache = (k, v)
        
        # Expand k, v if using grouped-query attention (keys and values are shared across query heads)
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply causal mask if needed
        if mask is not None:
            scores = scores + mask
        
        # Attention weights and context calculation
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to [batch_size, seq_length, hidden_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_dim)
        
        # Final projection
        return self.o_proj(context), new_kv_cache

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        num_kv_heads: int,
        ffn_dim: int,
        rope_config: Dict,
        max_seq_length: int
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            dim=dim, 
            num_heads=num_heads, 
            num_kv_heads=num_kv_heads,
            rope_config=rope_config,
            max_seq_length=max_seq_length
        )
        self.ffn = FeedForward(dim=dim, hidden_dim=ffn_dim)
        self.attention_norm = RMSNorm(dim, eps=1e-5)
        self.ffn_norm = RMSNorm(dim, eps=1e-5)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Self-attention with residual connection
        residual = x
        x = self.attention_norm(x)
        
        # Use KV cache for the layer if provided
        layer_kv_cache = None if kv_cache is None else kv_cache
        
        attn_output, new_kv_cache = self.self_attn(
            x, mask, layer_kv_cache, start_pos=start_pos
        )
        x = residual + attn_output
        
        # FFN with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        
        return x, new_kv_cache

class SimpleLLM(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        # Check for MPS (Apple Silicon) support first, then CUDA, then fall back to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.kv_cache = None
        self.load_weights(model_name)
        
    def load_weights(self, model_name: str) -> None:
        """
        Load weights from a LLaMA model using the transformers library.
        Implements weight caching to store in the current project directory.
        """
        # Create cache in the current directory
        cache_dir = os.path.join(os.getcwd(), ".simple_llm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate file paths
        model_slug = model_name.replace("/", "_")
        cache_path = os.path.join(cache_dir, f"{model_slug}_weights.pt")
        config_path = os.path.join(cache_dir, f"{model_slug}_config.json")
        
        # Check if weights are cached
        if os.path.exists(cache_path) and os.path.exists(config_path):
            print(f"Loading cached weights from {cache_path}")
            self.weights = torch.load(cache_path, map_location=self.device)
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            print(f"Downloading model {model_name}...")
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Extract and convert weights
            self.weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
            
            # Save config as dictionary
            config_dict = model.config.to_dict()
            self.config = config_dict
            
            # Cache weights and config
            torch.save(self.weights, cache_path)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f)
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Initialize model parameters
        self._init_model()
    
    def _init_model(self) -> None:
        """
        Initialize model parameters based on the config.
        """
        # Extract config parameters
        self.hidden_size = self.config.get("hidden_size")
        self.num_heads = self.config.get("num_attention_heads")
        self.num_kv_heads = self.config.get("num_key_value_heads", self.num_heads)
        self.num_layers = self.config.get("num_hidden_layers")
        self.vocab_size = self.config.get("vocab_size")
        self.ffn_dim = self.config.get("intermediate_size")
        self.max_seq_length = self.config.get("max_position_embeddings")
        
        # Configure RoPE
        self.rope_config = {
            "rope_base": self.config.get("rope_theta", 10000.0),
            "rope_freq": {
                "factor": self.config.get("rope_scaling", {}).get("factor", 1.0),
                "low_freq_factor": self.config.get("rope_scaling", {}).get("low_freq_factor", 1.0),
                "high_freq_factor": self.config.get("rope_scaling", {}).get("high_freq_factor", 1.0),
                "original_context_length": self.config.get("rope_scaling", {}).get("original_max_position_embeddings", self.max_seq_length)
            } if self.config.get("rope_scaling") else None
        }
        
        # Initialize model components
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_tokens.weight.data = self.weights["model.embed_tokens.weight"]
        
        # Initialize decoder layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                DecoderLayer(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    ffn_dim=self.ffn_dim,
                    rope_config=self.rope_config,
                    max_seq_length=self.max_seq_length
                )
            )
        
        # Initialize normalization and output layers
        self.norm = RMSNorm(self.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Load layer weights
        self._load_layer_weights()
        
    def _load_layer_weights(self) -> None:
        """
        Load weights into the model layers.
        """
        # Load weights for normalization at the end
        self.norm.weight.data = self.weights["model.norm.weight"]
        
        # Load weights for lm_head (might be tied with embeddings)
        if "lm_head.weight" in self.weights:
            self.lm_head.weight.data = self.weights["lm_head.weight"]
        else:
            # Weight tying (common in LLaMA models)
            self.lm_head.weight = self.embed_tokens.weight
            
        # Load weights for each layer
        for i, layer in enumerate(self.layers):
            # Attention weights
            layer.self_attn.q_proj.weight.data = self.weights[f"model.layers.{i}.self_attn.q_proj.weight"]
            layer.self_attn.k_proj.weight.data = self.weights[f"model.layers.{i}.self_attn.k_proj.weight"]
            layer.self_attn.v_proj.weight.data = self.weights[f"model.layers.{i}.self_attn.v_proj.weight"]
            layer.self_attn.o_proj.weight.data = self.weights[f"model.layers.{i}.self_attn.o_proj.weight"]
            
            # Normalization weights
            layer.attention_norm.weight.data = self.weights[f"model.layers.{i}.input_layernorm.weight"]
            layer.ffn_norm.weight.data = self.weights[f"model.layers.{i}.post_attention_layernorm.weight"]
            
            # Feed-forward weights
            layer.ffn.gate_proj.weight.data = self.weights[f"model.layers.{i}.mlp.gate_proj.weight"]
            layer.ffn.up_proj.weight.data = self.weights[f"model.layers.{i}.mlp.up_proj.weight"]
            layer.ffn.down_proj.weight.data = self.weights[f"model.layers.{i}.mlp.down_proj.weight"]
    
    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """
        Create a causal attention mask.
        """
        # Create mask where upper triangular part is -inf (including diagonal)
        mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
        return mask.to(self.device)
    
    def _init_kv_cache(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Initialize KV cache for faster generation.
        """
        kv_cache = []
        for _ in range(self.num_layers):
            # Initialize with the correct shapes but zero sequence length
            k_shape = (batch_size, self.num_kv_heads, 0, self.hidden_size // self.num_heads)
            v_shape = (batch_size, self.num_kv_heads, 0, self.hidden_size // self.num_heads)
            k_cache = torch.zeros(k_shape, device=self.device)
            v_cache = torch.zeros(v_shape, device=self.device)
            kv_cache.append((k_cache, v_cache))
        return kv_cache
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        use_kv_cache: bool = False,
        start_pos: int = 0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_length = input_ids.shape
        
        # Initialize or use existing KV cache
        if use_kv_cache:
            if self.kv_cache is None:
                self.kv_cache = self._init_kv_cache(batch_size)
            kv_cache = self.kv_cache
        else:
            kv_cache = None
        
        # Create causal mask - either for the whole sequence or for generation
        attention_mask = self._create_causal_mask(seq_length)
        
        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through each layer
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_kv_cache = None if kv_cache is None else kv_cache[i]
            hidden_states, new_layer_kv_cache = layer(
                hidden_states, 
                mask=attention_mask,
                kv_cache=layer_kv_cache,
                start_pos=start_pos
            )
            if use_kv_cache:
                new_kv_cache.append(new_layer_kv_cache)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        # Update KV cache if used
        if use_kv_cache:
            self.kv_cache = new_kv_cache
            return logits, new_kv_cache
        
        return logits
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 512, 
        temperature: float = 1.0, 
        top_k: int = 0,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate text given a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1.0 = more deterministic)
            top_k: Number of highest probability tokens to keep for sampling (0 = disabled)
            seed: Random seed for reproducibility
        
        Returns:
            generated_text: The generated text including the prompt
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Reset KV cache
        self.kv_cache = None
        
        # Tokenize input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Track generated tokens
        generated_ids = input_ids.clone()
        
        # First forward pass on the prompt
        with torch.no_grad():
            outputs = self.forward(input_ids, use_kv_cache=True, start_pos=0)
        
        # Start generating tokens one at a time
        for i in range(max_length):
            # Get position for the next token if using KV cache
            start_pos = generated_ids.shape[1]
            
            # Process only the last token with updated position
            curr_input_ids = generated_ids[:, -1:]
            
            # Forward pass with KV caching
            with torch.no_grad():
                outputs = self.forward(curr_input_ids, use_kv_cache=True, start_pos=start_pos)
                
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            # Get logits for the next token (last position)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k if specified
            if top_k > 0:
                top_values, top_indices = torch.topk(next_token_logits, k=top_k)
                mask = torch.zeros_like(next_token_logits).scatter_(1, top_indices, 1)
                next_token_logits.masked_fill_(mask == 0, float('-inf'))
            
            # Sample next token or pick highest probability token
            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to generated ids
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # Decode generated ids to text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text

