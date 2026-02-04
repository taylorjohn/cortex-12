"""
CORTEX-12 Latent Predictor: The "P" in JEPA

Transforms CORTEX-12 from Joint-Embedding Architecture (JEA)
to Joint-Embedding Predictive Architecture (JEPA).

Capabilities:
1. Predict masked/hidden semantic attributes
2. Imagine transformations (rotate, scale, recolor)
3. Plan action outcomes in latent space
4. World modeling without pixel generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictor(nn.Module):
    """
    Predicts target embeddings from context embeddings + conditioning
    
    This is the core "Predictor" module that makes CORTEX-12 a true JEPA.
    """
    
    def __init__(
        self,
        embedding_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        """
        Args:
            embedding_dim: Dimension of CORTEX-12 embeddings (128)
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Input projection
        self.context_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Conditioning token embeddings
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.transform_embedding = nn.Embedding(10, hidden_dim)  # For transform types
        
        # Transformer predictor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Per-axis predictors (optional, for fine-grained control)
        self.axis_predictors = nn.ModuleDict({
            'shape': nn.Linear(hidden_dim, 32),
            'size': nn.Linear(hidden_dim, 16),
            'color': nn.Linear(hidden_dim, 16),
            'material': nn.Linear(hidden_dim, 16),
        })
    
    def forward(
        self,
        context_embedding,
        target_mask=None,
        transform_id=None,
        predict_axes=None
    ):
        """
        Predict target embedding from context
        
        Args:
            context_embedding: (batch, 128) - Known embedding
            target_mask: (batch, 128) - Binary mask (1 = predict, 0 = use context)
            transform_id: (batch,) - Optional transform type (rotate, scale, etc.)
            predict_axes: List of axis names to predict
        
        Returns:
            predicted_embedding: (batch, 128) - Predicted full embedding
            per_axis_predictions: Dict of per-axis predictions
        """
        batch_size = context_embedding.size(0)
        
        # Project context to hidden space
        hidden = self.context_proj(context_embedding)  # (batch, hidden_dim)
        hidden = hidden.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Add mask token for prediction
        mask_tokens = self.mask_token.expand(batch_size, 1, -1)  # (batch, 1, hidden)
        
        # Optionally add transform conditioning
        if transform_id is not None:
            transform_emb = self.transform_embedding(transform_id).unsqueeze(1)
            seq = torch.cat([hidden, mask_tokens, transform_emb], dim=1)
        else:
            seq = torch.cat([hidden, mask_tokens], dim=1)
        
        # Apply transformer
        predicted = self.transformer(seq)  # (batch, seq_len, hidden)
        
        # Extract prediction (from mask token position)
        pred_hidden = predicted[:, 1, :]  # (batch, hidden)
        
        # Project to embedding space
        full_prediction = self.output_proj(pred_hidden)  # (batch, 128)
        
        # Apply mask if provided (blend context + prediction)
        if target_mask is not None:
            full_prediction = (
                context_embedding * (1 - target_mask) +
                full_prediction * target_mask
            )
        
        # Per-axis predictions (for fine-grained loss)
        per_axis = {}
        if predict_axes:
            for axis_name in predict_axes:
                if axis_name in self.axis_predictors:
                    per_axis[axis_name] = self.axis_predictors[axis_name](pred_hidden)
        
        return full_prediction, per_axis


class TransformationPredictor(nn.Module):
    """
    Specialized predictor for geometric transformations
    
    Predicts: "What would this object look like if rotated/scaled/recolored?"
    """
    
    TRANSFORMS = {
        'rotate_90': 0,
        'rotate_180': 1,
        'rotate_270': 2,
        'scale_up': 3,
        'scale_down': 4,
        'recolor_red': 5,
        'recolor_blue': 6,
        'recolor_green': 7,
        'flip_horizontal': 8,
        'flip_vertical': 9
    }
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.predictor = LatentPredictor(embedding_dim)
    
    def predict_transformation(self, embedding, transform_name):
        """
        Predict result of applying a transformation
        
        Args:
            embedding: (batch, 128) - Current object embedding
            transform_name: str - Name of transformation
        
        Returns:
            predicted: (batch, 128) - Predicted result embedding
        """
        transform_id = self.TRANSFORMS[transform_name]
        transform_tensor = torch.tensor(
            [transform_id] * embedding.size(0),
            device=embedding.device
        )
        
        predicted, _ = self.predictor(
            embedding,
            transform_id=transform_tensor
        )
        
        return predicted
    
    def plan_action_sequence(self, start_embedding, actions):
        """
        Plan a sequence of transformations
        
        Args:
            start_embedding: (batch, 128) - Starting state
            actions: List of transform names
        
        Returns:
            trajectory: List of (batch, 128) embeddings for each step
        """
        trajectory = [start_embedding]
        current = start_embedding
        
        for action in actions:
            next_state = self.predict_transformation(current, action)
            trajectory.append(next_state)
            current = next_state
        
        return trajectory


class MaskedPredictionTrainer:
    """
    Training procedure for latent prediction
    
    Trains the predictor to anticipate masked attributes or transformations.
    """
    
    def __init__(
        self,
        predictor,
        encoder,
        axis_dims,
        device='cpu'
    ):
        """
        Args:
            predictor: LatentPredictor module
            encoder: CORTEX-12 encoder (frozen or trainable)
            axis_dims: Dict of axis dimension ranges
            device: 'cpu', 'cuda', or 'mps'
        """
        self.predictor = predictor
        self.encoder = encoder
        self.axis_dims = axis_dims
        self.device = device
    
    def create_mask(self, embeddings, mask_axes):
        """
        Create mask for specified axes
        
        Args:
            embeddings: (batch, 128)
            mask_axes: List of axis names to mask (predict)
        
        Returns:
            mask: (batch, 128) binary mask
        """
        batch_size = embeddings.size(0)
        mask = torch.zeros(batch_size, 128, device=self.device)
        
        for axis_name in mask_axes:
            if axis_name in self.axis_dims:
                start, end = self.axis_dims[axis_name]
                mask[:, start:end+1] = 1.0
        
        return mask
    
    def train_step(self, images_context, images_target, mask_axes):
        """
        Single training step for masked prediction
        
        Args:
            images_context: Images to encode as context
            images_target: Target images (with masked attributes)
            mask_axes: Which axes to predict
        
        Returns:
            loss: Prediction loss
        """
        # Encode both
        with torch.no_grad():
            context_emb = self.encoder(images_context)
            target_emb = self.encoder(images_target)
        
        # Create mask
        mask = self.create_mask(context_emb, mask_axes)
        
        # Predict
        predicted, _ = self.predictor(context_emb, target_mask=mask)
        
        # Loss on masked regions only
        masked_pred = predicted * mask
        masked_target = target_emb * mask
        
        loss = F.mse_loss(masked_pred, masked_target)
        
        return loss


# Example: Transformation prediction task
if __name__ == "__main__":
    print("="*60)
    print("CORTEX-12 Latent Predictor Demo")
    print("="*60)
    
    # Create predictor
    predictor = TransformationPredictor(embedding_dim=128)
    
    # Simulate: Red circle → predict after rotation
    red_circle = torch.randn(1, 128)  # Fake embedding
    
    # Predict rotation
    predicted_rotated = predictor.predict_transformation(
        red_circle,
        'rotate_90'
    )
    
    print(f"\nOriginal embedding shape: {red_circle.shape}")
    print(f"Predicted (after rotate_90): {predicted_rotated.shape}")
    
    # Plan action sequence
    actions = ['rotate_90', 'scale_up', 'recolor_blue']
    trajectory = predictor.plan_action_sequence(red_circle, actions)
    
    print(f"\nAction sequence: {actions}")
    print(f"Trajectory length: {len(trajectory)} states")
    print("\nThis is world modeling in latent space! 🌍")
