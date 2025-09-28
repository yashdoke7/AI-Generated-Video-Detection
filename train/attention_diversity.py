import torch
import torch.nn as nn

class AttentionDiversityLoss(nn.Module):
    """
    Attention Diversity Loss for UNITE
    Encourages different attention heads to focus on different spatial regions
    """
    def __init__(self, diversity_weight=0.1):
        super().__init__()
        self.diversity_weight = diversity_weight
    
    def forward(self, transformer_output):
        """
        Args:
            transformer_output: [B, T, D] output from transformer encoder
        Returns:
            diversity_loss: scalar loss encouraging spatial attention diversity
        """
        batch_size, seq_len, hidden_dim = transformer_output.shape
        
        # Calculate attention weights across temporal sequence
        # This approximates spatial attention across video frames
        attention_weights = torch.norm(transformer_output, dim=-1)  # [B, T]
        attention_probs = torch.softmax(attention_weights, dim=-1)
        
        # Calculate concentration penalty (higher = more concentrated)
        # Shannon entropy-like measure for diversity
        concentration = torch.sum(attention_probs ** 2, dim=-1)  # [B]
        
        # Diversity loss: penalize high concentration
        diversity_loss = torch.mean(concentration)
        
        return self.diversity_weight * diversity_loss

class UNITEWithAttentionDiversity(nn.Module):
    """
    Enhanced TemporalTransformer with built-in attention diversity
    """
    def __init__(self, emb_dim=768, d_model=256, nhead=4, nlayers=2, max_len=64, diversity_weight=0.1):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(), 
            nn.Linear(128, 1)
        )
        
        # Attention diversity loss
        self.attention_diversity = AttentionDiversityLoss(diversity_weight)
        
    def forward(self, x, return_diversity_loss=False):
        """
        Args:
            x: [B, T, D] input embeddings
            return_diversity_loss: whether to return diversity loss
        """
        x = self.input_proj(x)
        transformer_out = self.encoder(x)  # [B, T, D]
        
        # Apply adapters if present
        if hasattr(self, 'adapters') and self.adapters is not None:
            for adapter in self.adapters:
                transformer_out = adapter(transformer_out)
        
        # Classification
        y = transformer_out.permute(0, 2, 1)  # [B, D, T]
        y = self.pool(y).squeeze(-1)  # [B, D]
        logits = self.head(y).squeeze(-1)  # [B]
        
        if return_diversity_loss:
            diversity_loss = self.attention_diversity(transformer_out)
            return logits, diversity_loss
        else:
            return logits
