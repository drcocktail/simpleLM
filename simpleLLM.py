import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleLLM(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.load_weights(model_name)
        self.past_key_values = None

    def load_weights(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.weights = {k: v.detach().requires_grad_(False) for k, v in model.state_dict().items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = model.config.__dict__  # Store full config as dictionary
        
        # Extract KV groups from config if available, otherwise use a reasonable default
        if 'num_key_value_heads' in self.config:
            self.num_kv_groups = self.config['num_key_value_heads']
        else:
            # Llama models typically use num_heads//4 KV groups
            self.num_kv_groups = self.config['num_attention_heads'] // 4
            print(f"KV groups not found in config, defaulting to {self.num_kv_groups}")

    def rms_norm(self, x, weight, eps=1e-5):
        # Updated epsilon to match Llama 3.2
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def rotary_embedding(self, q, k, position_ids):
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_len, num_kv_groups, head_dim]
            position_ids: Position IDs of shape [seq_len]
            
        Returns:
            Rotated query and key tensors with the same shape as inputs
        """
        # Get dimensions
        hidden_size = self.config['hidden_size']
        num_heads = self.config['num_attention_heads']
        head_dim = hidden_size // num_heads
        
        # Check and handle tensor dimensions
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        # Create position ids if not provided
        if position_ids.dim() > 1:
            position_ids = position_ids.squeeze()
        
        # Generate rotation matrices with scaled frequencies
        # Following Llama 3.2's RoPE implementation with a higher base value
        rope_base = 500000.0  # Higher base for extended context
        dim_half = head_dim // 2
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim_half, device=q.device).float() / dim_half))
        
        # Outer product of positions and frequencies
        # Results in shape [seq_len, dim_half]
        sincos = torch.outer(position_ids.float(), inv_freq)
        sin = sincos.sin()  # [seq_len, dim_half]
        cos = sincos.cos()  # [seq_len, dim_half]
        
        # Reshape for broadcasting
        sin = sin.view(1, seq_len, 1, dim_half)  # [1, seq_len, 1, dim_half]
        cos = cos.view(1, seq_len, 1, dim_half)  # [1, seq_len, 1, dim_half]
        
        # Apply rotary embeddings to query
        q_even = q[..., :dim_half]
        q_odd = q[..., dim_half:dim_half*2]
        q_rotated = torch.cat([
            q_even * cos - q_odd * sin,
            q_odd * cos + q_even * sin
        ], dim=-1)
        
        # Apply rotary embeddings to key
        k_even = k[..., :dim_half]
        k_odd = k[..., dim_half:dim_half*2]
        k_rotated = torch.cat([
            k_even * cos - k_odd * sin,
            k_odd * cos + k_even * sin
        ], dim=-1)
        
        return q_rotated, k_rotated

    def forward(self, input_ids, past_key_values=None):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Config parameters for attention
        hidden_size = self.config['hidden_size']
        num_heads = self.config['num_attention_heads']
        head_dim = hidden_size // num_heads
        group_size = num_heads // self.num_kv_groups
        
        # Initial token embeddings
        h = torch.nn.functional.embedding(input_ids, self.weights['model.embed_tokens.weight'])
        
        # Position ids for rotary embeddings
        if past_key_values is not None:
            position_offset = past_key_values[0][0].size(1)  # kv_seq_len from first layer's k
        else:
            position_offset = 0
        position_ids = torch.arange(seq_len, device=h.device) + position_offset
        
        new_kvs = []
        for i in range(self.config['num_hidden_layers']):
            # Layer normalization before attention
            h_norm = self.rms_norm(h, self.weights[f'model.layers.{i}.input_layernorm.weight'])
            
            # QKV projections
            q = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.q_proj.weight'].T)
            k = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.k_proj.weight'].T)
            v = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.v_proj.weight'].T)
            
            # Reshape for multi-head attention with grouped KV heads
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_groups, head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_groups, head_dim)
            
            # Apply rotary embeddings to q and k
            q, k = self.rotary_embedding(q, k, position_ids)
            
            # Append past key values if provided
            if past_key_values is not None:
                past_k = past_key_values[i][0]  # [batch, past_seq_len, num_kv_groups, head_dim]
                past_v = past_key_values[i][1]  # [batch, past_seq_len, num_kv_groups, head_dim]
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            
            # Store new key-values for next token generation
            new_kvs.append((k, v))
            
            # Expand KV heads to match query heads using repeat_interleave
            k_expanded = k.repeat_interleave(group_size, dim=2)  # [batch, seq, num_heads, head_dim]
            v_expanded = v.repeat_interleave(group_size, dim=2)  # [batch, seq, num_heads, head_dim]
            
            # Reshape tensors for attention computation
            q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            k = k_expanded.transpose(1, 2)  # [batch, num_heads, kv_seq_len, head_dim]
            v = v_expanded.transpose(1, 2)  # [batch, num_heads, kv_seq_len, head_dim]
            
            # Scaled dot-product attention
            # Compute attention scores
            scale = 1.0 / (head_dim ** 0.5)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [batch, num_heads, seq_len, kv_seq_len]
            
            # Apply causal mask
            kv_seq_len = k.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
                diagonal=1 + position_offset
            )
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Softmax to get attention weights
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention weights to values
            attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
            
            # Reshape back to combine heads
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            
            # Project attention output
            attn_output = torch.matmul(attn_output, self.weights[f'model.layers.{i}.self_attn.o_proj.weight'].T)
            
            # Residual connection
            h = h + attn_output
            
            # Second normalization for FFN
            h_norm = self.rms_norm(h, self.weights[f'model.layers.{i}.post_attention_layernorm.weight'])
            
            # FFN with SiLU activation (Swish)
            gate = torch.nn.functional.silu(torch.matmul(h_norm, self.weights[f'model.layers.{i}.mlp.gate_proj.weight'].T))
            up = torch.matmul(h_norm, self.weights[f'model.layers.{i}.mlp.up_proj.weight'].T)
            down = torch.matmul(gate * up, self.weights[f'model.layers.{i}.mlp.down_proj.weight'].T)
            
            # Residual connection for FFN
            h = h + down
        
        # Final normalization
        h = self.rms_norm(h, self.weights['model.norm.weight'])
        
        # Project to vocabulary
        if 'lm_head.weight' in self.weights:
            logits = torch.matmul(h, self.weights['lm_head.weight'].T)
        else:
            # Weight tying - use embedding weights
            logits = torch.matmul(h, self.weights['model.embed_tokens.weight'].T)
        
        return logits, new_kvs

    def generate(self, prompt, max_length=512, temperature=0.7, top_k=None):
        device = next(iter(self.weights.values())).device
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        past_key_values = None
        
        for _ in range(max_length):
            logits, past_key_values = self.forward(input_ids[:, -1:] if past_key_values else input_ids, past_key_values)
            
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                if top_k is not None:
                    top_probs, top_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def stream_generate(self, prompt, max_length=512, temperature=0.0, top_k=None):
        device = next(iter(self.weights.values())).device
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        past_key_values = None
        generated = []
        
        for _ in range(max_length):
            logits, past_key_values = self.forward(
                input_ids[:, -1:] if past_key_values else input_ids,
                past_key_values
            )
            
            # Sampling
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                if top_k is not None:
                    top_probs, top_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            
            # Decode and stream
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            new_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated.append(new_text)
            yield new_text
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        yield ''.join(generated)