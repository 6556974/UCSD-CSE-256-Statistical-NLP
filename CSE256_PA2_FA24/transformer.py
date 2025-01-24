# transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads == embed_dim), "Embedding dimension must be divisible by number of heads"

        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, T, C = x.size()
        # Compute q, k, v matrices
        qkv = self.qkv_linear(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, T, head_dim)

        # Compute scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # Shape: (B, num_heads, T, T)

        # Apply mask (if provided)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)  # Apply softmax along the last dimension
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)  # Shape: (B, T, embed_dim)
        return self.out_linear(out), attn_weights


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T = x.size()
        token_emb = self.token_embedding(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_maps.append(attn_weights)
        return self.norm(x), attn_maps


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.vocab_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        # Ensure input is of the correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        elif x.dim() != 2:
            raise ValueError(f"Expected input tensor of shape (B, T), but got shape {x.shape}")

        B, T = x.size()
        token_emb = self.token_embedding(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb

        # Create a mask to prevent the decoder from attending to future tokens
        mask = torch.tril(torch.ones(T, T, device=x.device)).expand(B, 1, T, T)

        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_maps.append(attn_weights)

        x = self.norm(x)
        logits = self.vocab_projection(x)

        # If targets are provided, compute the loss
        if targets is not None:
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)  # Add batch dimension if missing
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return loss  # Directly return the loss if targets are provided

        return logits, attn_maps

    def generate(self, idx, max_new_tokens):
        """ Generate text by predicting next token sequentially """
        for _ in range(max_new_tokens):
            if idx.dim() == 1:
                idx = idx.unsqueeze(0)  # Add batch dimension if missing
            idx_cond = idx[:, -self.position_embedding.num_embeddings:]  # Limit to max_len
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Get the logits for the last token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the probability distribution
            idx = torch.cat((idx, idx_next), dim=1)
        return idx




class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Use mean pooling to get the representation
        x = x.mean(dim=1)
        return self.fc(x)




class MultiHeadAttention2(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1, use_alibi=False):
        super(MultiHeadAttention2, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.use_alibi = use_alibi

        assert (self.head_dim * num_heads == embed_dim), "Embedding dimension must be divisible by number of heads"

        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, T, C = x.size()
        # Compute q, k, v matrices
        qkv = self.qkv_linear(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, T, head_dim)

        # Compute scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # Shape: (B, num_heads, T, T)

        # Apply AliBi bias if specified
        if self.use_alibi:
            bias = torch.arange(T, device=x.device).unsqueeze(0) - torch.arange(T, device=x.device).unsqueeze(1)
            bias = -bias.abs().float()  # Negative distance to favor closer tokens
            bias = bias.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            attn_weights = attn_weights + bias

        # Apply mask (if provided)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)  # Apply softmax along the last dimension
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)  # Shape: (B, T, embed_dim)
        return self.out_linear(out), attn_weights


class FeedForward2(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward2, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class EncoderBlock2(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, use_alibi=False):
        super(EncoderBlock2, self).__init__()
        self.attention = MultiHeadAttention2(num_heads, embed_dim, dropout, use_alibi=use_alibi)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward2(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights

class TransformerEncoder2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout=0.1, use_alibi=False):
        super(TransformerEncoder2, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([EncoderBlock2(embed_dim, num_heads, ff_dim, dropout, use_alibi=use_alibi) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        B, T = x.size()
        token_emb = self.token_embedding(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_maps.append(attn_weights)
        return self.norm(x), attn_maps


class DisentangledMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1):
        super(DisentangledMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads == embed_dim), "Embedding dimension must be divisible by number of heads"

        self.content_linear = nn.Linear(embed_dim, 2 * embed_dim)
        self.position_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x, position_embedding, mask=None):
        B, T, C = x.size()
        # Compute content-based query and key-value matrices
        content_qk = self.content_linear(x).reshape(B, T, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        content_q, content_k = content_qk[0], content_qk[1]  # Each has shape (B, num_heads, T, head_dim)

        # Compute position-based key matrix
        position_k = self.position_linear(position_embedding).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2,
                                                                                                                   1, 3)

        # Compute disentangled attention (content-to-content and content-to-position)
        attn_weights_content = (content_q @ content_k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        attn_weights_position = (content_q @ position_k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)

        # Combine attention weights
        attn_weights = attn_weights_content + attn_weights_position

        # Apply mask (if provided)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute value matrix
        value = content_k  # Use the content key as value in this implementation

        # Compute attention output
        out = (attn_weights @ value).transpose(1, 2).reshape(B, T, C)  # Shape: (B, T, embed_dim)
        return self.out_linear(out), attn_weights


class EncoderBlock3(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderBlock3, self).__init__()
        self.attention = DisentangledMultiHeadAttention(num_heads, embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, position_embedding, mask=None):
        attn_out, attn_weights = self.attention(x, position_embedding, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights


class TransformerEncoder3(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout=0.1):
        super(TransformerEncoder3, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([EncoderBlock3(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        B, T = x.size()
        token_emb = self.token_embedding(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, position_emb, mask)
            attn_maps.append(attn_weights)
        return self.norm(x), attn_maps
