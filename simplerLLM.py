import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def precompute_rope_params(head_dim, theta_base=10_000, freq_config=None, device="cpu"):
    # From standalone-llama32.ipynb
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    if freq_config is not None:
        low_freq_wavelen = freq_config["original_max_position_embeddings"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_max_position_embeddings"] / freq_config["high_freq_factor"]
        wavelen = 2 * torch.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq)
        smooth_factor = (freq_config["original_max_position_embeddings"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    
    return inv_freq

def apply_rope(x, inv_freq, position_ids):
    # From standalone-llama32.ipynb, adapted for our class structure
    head_dim = x.size(-1)
    
    sinusoid = torch.outer(position_ids.to(inv_freq.dtype), inv_freq)
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0).unsqueeze(0)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0).unsqueeze(0)

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos.to(x.dtype)) + (x_rotated * sin.to(x.dtype))

class SimpleLLM(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Set device with MPS priority
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (no GPU acceleration available)")
        
        self.load_model(model_name)

    def load_model(self, model_name):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config = config.to_dict()

        # Use float32 for MPS to ensure stability, as bfloat16 is not fully supported
        self.dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )

        print(f"Moving model weights to {self.device}...")
        self.weights = {k: v.detach().to(self.device) for k, v in model.state_dict().items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.num_kv_groups = self.config.get('num_key_value_heads', self.config['num_attention_heads'] // self.config.get('num_attention_heads', 1))

        # Pre-compute RoPE frequencies
        head_dim = self.config['hidden_size'] // self.config['num_attention_heads']
        rope_config = self.config.get("rope_scaling")
        if rope_config:
             # HuggingFace configs call it 'original_max_position_embeddings'
             rope_config['original_max_position_embeddings'] = rope_config.pop('original_context_length', 8192)
        
        self.inv_freq = precompute_rope_params(
            head_dim=head_dim,
            theta_base=self.config.get("rope_theta", 500000.0),
            freq_config=rope_config,
            device=self.device
        ).to(self.device).to(self.dtype)

        print(f"Model loaded successfully on {self.device}")

    def rms_norm(self, x, weight, eps=1e-5):
        # Cast to float32 for the calculation to avoid instability
        input_dtype = x.dtype
        x = x.to(torch.float32)
        weight = weight.to(torch.float32)
        
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        
        return (x * weight).to(input_dtype)

    def forward(self, input_ids, past_key_values=None):
        batch_size, seq_len = input_ids.size()
        is_prompt_processing = (past_key_values is None)

        if is_prompt_processing:
            self._absolute_pos = 0

        hidden_size = self.config['hidden_size']
        num_heads = self.config['num_attention_heads']
        head_dim = hidden_size // num_heads
        num_layers = self.config['num_hidden_layers']
        
        # Ensure embedding is done in float32
        h = F.embedding(input_ids, self.weights['model.embed_tokens.weight']).to(self.dtype)
        
        position_ids = torch.arange(self._absolute_pos, self._absolute_pos + seq_len, device=self.device, dtype=torch.long)
        self._absolute_pos += seq_len
        
        new_kvs = []
        for i in range(num_layers):
            # Force calculations in float32
            h_norm = self.rms_norm(h, self.weights[f'model.layers.{i}.input_layernorm.weight'])
            
            q = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.q_proj.weight'].T)
            k = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.k_proj.weight'].T)
            v = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.v_proj.weight'].T)
            
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_kv_groups, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_kv_groups, head_dim).transpose(1, 2)

            q = apply_rope(q, self.inv_freq, position_ids)
            k = apply_rope(k, self.inv_freq, position_ids)
            
            if not is_prompt_processing:
                past_k, past_v = past_key_values[i]
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            new_kvs.append((k.clone(), v.clone()))
            
            if self.num_kv_groups != num_heads:
                k = k.repeat_interleave(num_heads // self.num_kv_groups, dim=1)
                v = v.repeat_interleave(num_heads // self.num_kv_groups, dim=1)
            
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            attn_output = torch.matmul(attn_output, self.weights[f'model.layers.{i}.self_attn.o_proj.weight'].T)
            
            h = h + attn_output
            
            h_residual = h
            h_norm = self.rms_norm(h, self.weights[f'model.layers.{i}.post_attention_layernorm.weight'])
            
            gate = F.silu(torch.matmul(h_norm, self.weights[f'model.layers.{i}.mlp.gate_proj.weight'].T))
            up = torch.matmul(h_norm, self.weights[f'model.layers.{i}.mlp.up_proj.weight'].T)
            h = h_residual + torch.matmul(gate * up, self.weights[f'model.layers.{i}.mlp.down_proj.weight'].T)
        
        h = self.rms_norm(h, self.weights['model.norm.weight'])
        lm_weight = self.weights.get('lm_head.weight', self.weights['model.embed_tokens.weight'])
        logits = torch.matmul(h, lm_weight.T)
        
        return logits.to(torch.float32), new_kvs

    def generate(self, prompt, temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.2, max_new_tokens=256):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        generated_ids_tensor = torch.tensor([[]], dtype=torch.long, device=self.device)
        past_key_values = None
        
        # safety_limit = self.config.get('max_position_embeddings', 8192)
        
        with torch.inference_mode():
            for i in range(max_new_tokens):

                current_input = input_ids if past_key_values is None else input_ids[:, -1:]
                logits, past_key_values = self.forward(current_input, past_key_values)
                next_token_logits = logits[:, -1, :]
                
                if repetition_penalty != 1.0 and generated_ids_tensor.shape[1] > 0:
                    unique_tokens = torch.unique(generated_ids_tensor[0, -64:]) # check last 64 tokens
                    next_token_logits[0, unique_tokens] /= repetition_penalty

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    
                    if top_k is not None:
                        # In-place top-k filtering
                        v, _ = torch.topk(next_token_logits, top_k)
                        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[0, sorted_indices_to_remove[0]]
                        next_token_logits[:, indices_to_remove] = float('-inf')

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated_ids_tensor = torch.cat([generated_ids_tensor, next_token], dim=-1)
                token_id = next_token.item()
                
                yield self.tokenizer.decode([token_id])

                if token_id == self.tokenizer.eos_token_id:
                    break

                # No KV cache context management for now, to keep it simple and match notebook
    
    def get_memory_usage(self):
        """Get current memory usage on MPS device."""
        if self.device.type == 'mps':
            try:
                return torch.mps.current_allocated_memory() / 1024**3  # GB
            except:
                return "Memory tracking not available"
        elif self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            return "CPU mode - no GPU memory tracking"