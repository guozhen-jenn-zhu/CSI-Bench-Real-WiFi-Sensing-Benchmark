import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MLPClassifier(nn.Module):
    """Multi-layer Perceptron for WiFi sensing"""
    def __init__(self, win_len=500, feature_size=232, num_classes=2):
        super(MLPClassifier, self).__init__()
        # Calculate input size but limit it to prevent memory issues
        input_size = min(win_len * feature_size, 10000)
        
        self.win_len = win_len
        self.feature_size = feature_size
        self.num_classes = num_classes
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def get_init_params(self):
        """Return the initialization parameters to support model cloning for few-shot learning"""
        return {
            'win_len': self.win_len,
            'feature_size': self.feature_size,
            'num_classes': self.num_classes
        }
        
    def forward(self, x):
        # Flatten input: [batch, channels, win_len, feature_size] -> [batch, win_len*feature_size]
        x = x.view(x.size(0), -1)
        # Limit input size if needed
        if x.size(1) > 10000:
            x = x[:, :10000]
        return self.fc(x)

class LSTMClassifier(nn.Module):
    """LSTM model for WiFi sensing"""
    def __init__(self, feature_size=232, hidden_size=256, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def get_init_params(self):
        """Return the initialization parameters to support model cloning for few-shot learning"""
        return {
            'feature_size': self.feature_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout
        }
        
    def forward(self, x):
        # Input shape: [batch, channels, win_len, feature_size]
        # LSTM expects: [batch, win_len, feature_size]
        x = x.squeeze(1)  # Remove channel dimension
        
        # Check if dimensions are transposed (win_len and feature_size swapped)
        if x.shape[2] == self.feature_size:
            # If feature_size is in the last dimension, we're good
            pass
        else:
            # If feature_size is not in the last dimension, transpose
            x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the final hidden state from both directions
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Classification
        out = self.fc(hidden_cat)
        return out

class ResNet18Classifier(nn.Module):
    """Modified ResNet-18 for WiFi sensing"""
    def __init__(self, win_len=500, feature_size=232, num_classes=2, in_channels=1):
        super(ResNet18Classifier, self).__init__()
        
        # Save the parameters
        self.win_len = win_len
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Load pretrained ResNet-18
        self.resnet = resnet18(pretrained=False)
        
        # Modify first conv layer to accept single channel
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fc layer
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def get_init_params(self):
        """Return the initialization parameters to support model cloning for few-shot learning"""
        return {
            'win_len': self.win_len,
            'feature_size': self.feature_size,
            'num_classes': self.num_classes,
            'in_channels': self.in_channels
        }
        
    def forward(self, x):
        # ResNet forward pass
        return self.resnet(x)

class TransformerClassifier(nn.Module):
    """Transformer model for WiFi sensing"""
    def __init__(self, feature_size=98, d_model=256, nhead=8, 
                 num_layers=4, dropout=0.1, num_classes=2, win_len=None):
        super(TransformerClassifier, self).__init__()
        
        self.feature_size = feature_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.win_len = win_len
        
        # Create input projection with the given feature_size as a placeholder
        # It will be replaced in the forward method if needed
        self.input_proj = nn.Linear(feature_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes)
        )
    
    def get_init_params(self):
        """Return the initialization parameters to support model cloning for few-shot learning"""
        return {
            'feature_size': self.feature_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'num_classes': self.num_classes,
            'win_len': self.win_len
        }
    
    def _get_state_dict(self):
        """Get state_dict but exclude dynamically created input_proj if it's different from the initial one"""
        state_dict = super(TransformerClassifier, self).state_dict()
        # If the input_proj has been dynamically created with a different size, exclude it
        if hasattr(self.input_proj, 'in_features') and self.input_proj.in_features != self.feature_size:
            # Remove input_proj keys since they'll be dynamically created based on actual input
            keys_to_remove = [k for k in state_dict.keys() if k.startswith('input_proj')]
            for key in keys_to_remove:
                del state_dict[key]
        return state_dict
    
    def state_dict(self, *args, **kwargs):
        """Override state_dict to handle dynamically created input_proj"""
        return self._get_state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle potentially missing input_proj"""
        # Check for input_proj keys
        input_proj_keys = [k for k in state_dict.keys() if k.startswith('input_proj')]
        
        # If the state_dict has input_proj but with a different size, skip those keys
        if not input_proj_keys:
            # Create a new state dict excluding input_proj keys from current model
            model_state_dict = self.state_dict()
            # Load the rest of the state dict normally
            return super(TransformerClassifier, self).load_state_dict(state_dict, strict=False)
        else:
            # Try normal loading
            try:
                return super(TransformerClassifier, self).load_state_dict(state_dict, strict=strict)
            except RuntimeError as e:
                # If error due to input_proj, try loading without strict
                if "input_proj" in str(e):
                    return super(TransformerClassifier, self).load_state_dict(state_dict, strict=False)
                raise e
        
    def forward(self, x):
        # Input shape: [batch, channels, win_len, feature_size]
        # Transform to: [batch, win_len, feature_size]
        x = x.squeeze(1)  # Remove channel dimension
        
        # Determine the actual feature size from the input tensor
        actual_feature_size = x.shape[-1]
        if self.input_proj.in_features != actual_feature_size:
            # Create a new linear layer with the correct input size
            self.input_proj = nn.Linear(actual_feature_size, self.d_model).to(x.device)
        
        # Project to d_model dimensions
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Global average pooling over sequence length
        x = x.mean(dim=1)
        
        # Classification
        return self.classifier(x)

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



# --- Patch Embedding and Position Embedding ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=(4, 4), emb_dim=128, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(emb_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class ViTEmbedding(nn.Module):
    def __init__(self, win_len, feature_size, emb_dim=128, in_channels=1):
        super().__init__()
        patch_h = max(1, feature_size // 10)
        patch_w = max(1, win_len // 10)
        self.embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=(patch_h, patch_w),
            emb_dim=emb_dim
        )
        self.num_patches = (win_len // patch_w) * (feature_size // patch_h)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        if seq_len != self.pos_embedding.size(1):
            pos_embed = self.pos_embedding.transpose(1, 2)
            pos_embed = F.interpolate(pos_embed, size=seq_len, mode='linear')
            pos_embed = pos_embed.transpose(1, 2)
            x = x + pos_embed
        else:
            x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, E)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class ViTClassifier(nn.Module):
    def __init__(self, win_len=500, feature_size=232, in_channels=1, emb_dim=128, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1, num_classes=2):
        super().__init__()
        self.embedding = ViTEmbedding(win_len, feature_size, emb_dim, in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.encoder = TransformerEncoder(
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])

# -------------- New Time-Series Transformer Models --------------

class PatchTST(nn.Module):
    """
    Patch Time Series Transformer for time series classification.
    
    References:
    - Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    """
    def __init__(
        self, 
        win_len=500, 
        feature_size=232, 
        patch_len=16,
        stride=8,
        emb_dim=128, 
        depth=4, 
        num_heads=4, 
        dropout=0.1, 
        fc_dropout=0.3,
        head_dropout=0.2,
        num_classes=2,
        pool='cls',  # 'cls' or 'mean'
        in_channels=1
    ):
        super().__init__()
        # Model parameters
        self.win_len = win_len
        self.feature_size = feature_size
        self.patch_len = patch_len
        self.stride = stride
        self.emb_dim = emb_dim
        self.pool = pool
        
        # Calculate number of patches
        self.num_patches = (win_len - patch_len) // stride + 1
        
        # Patch embedding
        self.patch_embedding = nn.Conv1d(
            in_channels=feature_size,
            out_channels=emb_dim,
            kernel_size=patch_len,
            stride=stride
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + (1 if pool == 'cls' else 0), emb_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # CLS token (optional)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim)) if pool == 'cls' else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Dropout after embedding
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=4 * emb_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Layer norm
        self.norm = nn.LayerNorm(emb_dim)
        
        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(emb_dim, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [batch, channels, win_len, feature_size]
        batch_size = x.shape[0]
        
        # Handle input dimensions
        if len(x.shape) == 4:
            x = x.squeeze(1)  # Remove channel dimension if present
        
        # Ensure correct shape [batch, feature_size, win_len]
        if x.shape[1] == self.win_len and x.shape[2] == self.feature_size:
            # Swap dimensions if necessary
            x = x.transpose(1, 2)
        
        # Apply patch embedding [batch, feature_size, win_len] -> [batch, emb_dim, num_patches]
        x = self.patch_embedding(x)
        
        # Transpose to [batch, num_patches, emb_dim]
        x = x.transpose(1, 2)
        
        # Add CLS token if using 'cls' pooling
        if self.pool == 'cls' and self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Pool features according to strategy
        if self.pool == 'cls':
            x = x[:, 0]  # Take CLS token representation
        else:  # 'mean'
            x = x.mean(dim=1)  # Mean pooling over patches
        
        # Classification head
        return self.head(x)


class TimesFormer1D(nn.Module):
    """
    TimesFormer-1D: Transformer for time series classification with divided space-time attention.
    
    References:
    - Adapted from "Is Space-Time Attention All You Need for Video Understanding?"
    """
    def __init__(
        self,
        win_len=500,
        feature_size=232,
        patch_size=4,
        emb_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1,
        head_dropout=0.2,
        num_classes=2,
        in_channels=1
    ):
        super().__init__()
        self.win_len = win_len
        self.feature_size = feature_size
        self.emb_dim = emb_dim
        
        # Calculate number of patches (win_len must be divisible by patch_size)
        assert win_len % patch_size == 0, "win_len must be divisible by patch_size"
        self.num_patches = win_len // patch_size
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv1d(feature_size, emb_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e n -> b n e')
        )
        
        # Position embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Temporal and feature transformers
        self.blocks = nn.ModuleList([
            TimesFormerBlock(
                dim=emb_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(emb_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(head_dropout),
            nn.Linear(emb_dim, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [batch, channels, win_len, feature_size]
        batch_size = x.shape[0]
        
        # Handle input dimensions
        if len(x.shape) == 4:
            x = x.squeeze(1)  # Remove channel dimension if present
        
        # Ensure shape is [batch, feature_size, win_len]
        if x.shape[1] == self.win_len and x.shape[2] == self.feature_size:
            x = x.transpose(1, 2)
            
        # Patch embedding: [batch, feature_size, win_len] -> [batch, num_patches, emb_dim]
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks with divided attention
        for block in self.blocks:
            x = block(x)
            
        # Apply layer norm
        x = self.norm(x)
        
        # Use CLS token for classification
        x = x[:, 0]
        
        # Classification head
        return self.head(x)


class Rearrange(nn.Module):
    """Helper module for rearranging tensor dimensions."""
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern
        
    def forward(self, x):
        return x.permute(0, 2, 1)


class TimesFormerBlock(nn.Module):
    """TimesFormer block with divided temporal and feature attention."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        
        # Temporal attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn_temporal = MultiHeadAttention(dim, num_heads, attn_dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feature attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn_feature = MultiHeadAttention(dim, num_heads, attn_dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.norm3 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Temporal attention
        x = x + self.dropout1(self.attn_temporal(self.norm1(x)))
        
        # Feature attention
        x = x + self.dropout2(self.attn_feature(self.norm2(x)))
        
        # Feed-forward
        x = x + self.mlp(self.norm3(x))
        
        return x
