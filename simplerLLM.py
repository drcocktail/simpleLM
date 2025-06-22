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
        self.num_kv_groups = self.config.get('num_key_value_heads', self.config['num_attention_heads'] // 4)
        if 'num_key_value_heads' not in self.config:
            print(f"KV groups not found in config, defaulting to {self.num_kv_groups}")

    def rms_norm(self, x, weight, eps=1e-5):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def rotary_embedding(self, q, k, position_ids):
        """Apply rotary position embeddings to query and key tensors."""
        # Get dimensions and prepare tensors
        head_dim = q.size(-1)
        dim_half = head_dim // 2
        
        # Generate rotation matrices with scaled frequencies
        rope_base = 500000.0
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim_half, device=q.device).float() / dim_half))
        
        # Simplify position_ids handling
        if position_ids.dim() > 1:
            position_ids = position_ids.squeeze()
            
        # Calculate sin and cos values
        sincos = torch.outer(position_ids.float(), inv_freq)
        sin, cos = sincos.sin().unsqueeze(0).unsqueeze(2), sincos.cos().unsqueeze(0).unsqueeze(2)
        
        # Define a helper function to apply rotation to q or k
        def apply_rotation(x):
            x_even, x_odd = x[..., :dim_half], x[..., dim_half:dim_half*2]
            return torch.cat([
                x_even * cos - x_odd * sin,
                x_odd * cos + x_even * sin
            ], dim=-1)
            
        return apply_rotation(q), apply_rotation(k)

    def forward(self, input_ids, past_key_values=None):
        batch_size, seq_len = input_ids.size()
        
        # Extract config parameters
        hidden_size = self.config['hidden_size']
        num_heads = self.config['num_attention_heads']
        head_dim = hidden_size // num_heads
        num_layers = self.config['num_hidden_layers']
        group_size = num_heads // self.num_kv_groups
        
        # Initial token embeddings
        h = torch.nn.functional.embedding(input_ids, self.weights['model.embed_tokens.weight'])
        
        # Position ids for rotary embeddings
        position_offset = 0 if past_key_values is None else past_key_values[0][0].size(1)
        position_ids = torch.arange(seq_len, device=h.device) + position_offset
        
        # Process each layer
        new_kvs = []
        for i in range(num_layers):
            # Layer normalization and QKV projections
            h_norm = self.rms_norm(h, self.weights[f'model.layers.{i}.input_layernorm.weight'])
            q = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.q_proj.weight'].T)
            k = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.k_proj.weight'].T)
            v = torch.matmul(h_norm, self.weights[f'model.layers.{i}.self_attn.v_proj.weight'].T)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_groups, head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_groups, head_dim)
            
            # Apply rotary embeddings
            q, k = self.rotary_embedding(q, k, position_ids)
            
            # Handle past key values
            if past_key_values is not None:
                k = torch.cat([past_key_values[i][0], k], dim=1)
                v = torch.cat([past_key_values[i][1], v], dim=1)
            
            # Store new key-values for next token generation
            new_kvs.append((k, v))
            
            # Calculate attention scores and apply attention
            kv_seq_len = k.size(1)
            
            # Expand KV heads and prepare tensors for attention computation
            k_exp = k.repeat_interleave(group_size, dim=2).transpose(1, 2)  # [batch, num_heads, kv_seq_len, head_dim]
            v_exp = v.repeat_interleave(group_size, dim=2).transpose(1, 2)  # [batch, num_heads, kv_seq_len, head_dim]
            q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            
            # Compute attention with causal mask
            attn_scores = torch.matmul(q, k_exp.transpose(-1, -2)) / (head_dim ** 0.5)
            causal_mask = torch.triu(torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool), 
                                     diagonal=1 + position_offset)
            attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Apply attention weights and project
            attn_output = torch.matmul(torch.softmax(attn_scores, dim=-1), v_exp)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
            attn_output = torch.matmul(attn_output, self.weights[f'model.layers.{i}.self_attn.o_proj.weight'].T)
            
            # Residual connection and FFN
            h = h + attn_output
            h_norm = self.rms_norm(h, self.weights[f'model.layers.{i}.post_attention_layernorm.weight'])
            
            # FFN computation
            gate = torch.nn.functional.silu(torch.matmul(h_norm, self.weights[f'model.layers.{i}.mlp.gate_proj.weight'].T))
            up = torch.matmul(h_norm, self.weights[f'model.layers.{i}.mlp.up_proj.weight'].T)
            h = h + torch.matmul(gate * up, self.weights[f'model.layers.{i}.mlp.down_proj.weight'].T)
        
        # Final normalization and projection to vocabulary
        h = self.rms_norm(h, self.weights['model.norm.weight'])
        lm_weight = self.weights.get('lm_head.weight', self.weights['model.embed_tokens.weight'])
        logits = torch.matmul(h, lm_weight.T)
        
        return logits, new_kvs

    def generate(self, prompt, max_length=512, temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.2):
        """
        Generate text using improved sampling strategies to prevent repetition collapse.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Controls randomness (higher = more random)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Limits sampling to top k tokens (use None to disable)
            repetition_penalty: Penalty applied to previously generated tokens (1.0 = no penalty)
        
        Returns:
            Generated text including the prompt
        """
        device = next(iter(self.weights.values())).device
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        past_key_values = None
        
        # Track generated tokens for repetition penalty
        generated_tokens = input_ids[0].tolist()
        
        for _ in range(max_length):
            # Forward pass with past key values
            current_input = input_ids[:, -1:] if past_key_values else input_ids
            logits, past_key_values = self.forward(current_input, past_key_values)
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty to already generated tokens
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (not recommended for creative text)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Add the sampled token to the input_ids and track it
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens.append(next_token.item())
            
            # Stop if we reached the EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Early stopping for repetition (detect consecutive repeated sequences)
            n = 8  # Length of sequence to check for repetition
            if len(generated_tokens) > n*3 and generated_tokens[-n:] == generated_tokens[-2*n:-n]:
                # If we detect the same sequence repeating, stop generation
                print("Detected repetition loop, stopping generation")
                break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)