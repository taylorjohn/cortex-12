"""
CORTEX-12 v13: SIGReg Loss (LeJEPA-style)
Sketched Isotropic Gaussian Regularization

This replaces complex contrastive loss with mathematical guarantees
against semantic axis collapse. Based on LeJEPA (LeCun et al., Nov 2025).

Key Innovation:
- Forces covariance matrix toward identity
- Diagonal = 1 (variance maintenance)
- Off-diagonal = 0 (decorrelation/independence)
- Naturally enforces disentanglement without explicit axis masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization Loss
    
    Forces embeddings to have:
    1. Unit variance per dimension (diagonal = 1)
    2. Zero correlation between dimensions (off-diagonal = 0)
    
    This mathematically prevents semantic axis collapse.
    """
    
    def __init__(
        self,
        lambda_variance=1.0,
        lambda_covariance=1.0,
        eps=1e-4
    ):
        """
        Args:
            lambda_variance: Weight for variance regularization
            lambda_covariance: Weight for covariance regularization
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.lambda_variance = lambda_variance
        self.lambda_covariance = lambda_covariance
        self.eps = eps
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, embedding_dim)
        
        Returns:
            loss: SIGReg loss value
            stats: Dictionary with variance and covariance loss
        """
        batch_size, dim = embeddings.shape
        
        # Center embeddings (zero mean)
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        # cov = E[x x^T] where x is centered
        cov = (embeddings_centered.T @ embeddings_centered) / (batch_size - 1)
        
        # Variance loss: Force diagonal to 1
        # This maintains unit variance per dimension
        diagonal = torch.diagonal(cov)
        variance_loss = torch.mean((diagonal - 1.0) ** 2)
        
        # Covariance loss: Force off-diagonal to 0
        # This decorrelates dimensions (prevents collapse)
        # Sum all elements, subtract diagonal contribution
        covariance_loss = (cov ** 2).sum() - (diagonal ** 2).sum()
        covariance_loss = covariance_loss / (dim * dim - dim)  # Normalize
        
        # Total loss
        total_loss = (
            self.lambda_variance * variance_loss +
            self.lambda_covariance * covariance_loss
        )
        
        # Statistics for monitoring
        stats = {
            'variance_loss': variance_loss.item(),
            'covariance_loss': covariance_loss.item(),
            'total_loss': total_loss.item(),
            'mean_variance': diagonal.mean().item(),
            'max_off_diagonal': (cov - torch.diag(diagonal)).abs().max().item()
        }
        
        return total_loss, stats


class PerAxisSIGReg(nn.Module):
    """
    Apply SIGReg independently to each semantic axis
    
    This maintains axis-specific structure while preventing collapse.
    """
    
    def __init__(
        self,
        axis_dims,
        lambda_variance=1.0,
        lambda_covariance=1.0
    ):
        """
        Args:
            axis_dims: Dict mapping axis names to (start, end) dimension indices
                      e.g., {'shape': (0, 31), 'color': (64, 79)}
            lambda_variance: Variance regularization weight
            lambda_covariance: Covariance regularization weight
        """
        super().__init__()
        self.axis_dims = axis_dims
        self.sigreg = SIGRegLoss(lambda_variance, lambda_covariance)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, 128) full CORTEX-12 embedding
        
        Returns:
            total_loss: Sum of SIGReg losses across all axes
            per_axis_stats: Dictionary of statistics per axis
        """
        total_loss = 0.0
        per_axis_stats = {}
        
        for axis_name, (start, end) in self.axis_dims.items():
            # Extract axis-specific embeddings
            axis_emb = embeddings[:, start:end+1]
            
            # Apply SIGReg
            axis_loss, stats = self.sigreg(axis_emb)
            
            total_loss += axis_loss
            per_axis_stats[axis_name] = stats
        
        return total_loss, per_axis_stats


class HybridSIGRegContrastive(nn.Module):
    """
    Hybrid loss combining SIGReg (for collapse prevention) with
    contrastive loss (for semantic discrimination).
    
    Best of both worlds:
    - SIGReg prevents axis collapse
    - Contrastive ensures semantic clustering
    """
    
    def __init__(
        self,
        axis_dims,
        temperature=0.1,
        lambda_sigreg=0.5,
        lambda_contrastive=0.5
    ):
        super().__init__()
        self.axis_dims = axis_dims
        self.temperature = temperature
        self.lambda_sigreg = lambda_sigreg
        self.lambda_contrastive = lambda_contrastive
        self.sigreg = PerAxisSIGReg(axis_dims)
    
    def contrastive_loss(self, embeddings, labels, axis_name):
        """
        Standard contrastive loss for semantic discrimination
        """
        start, end = self.axis_dims[axis_name]
        axis_emb = embeddings[:, start:end+1]
        axis_emb = F.normalize(axis_emb, dim=1)
        
        # Similarity matrix
        sim = torch.mm(axis_emb, axis_emb.t()) / self.temperature
        
        # Same-class mask
        same_mask = (labels == labels.unsqueeze(1)).float()
        same_mask.fill_diagonal_(0)  # Remove self-similarity
        
        # InfoNCE-style loss
        exp_sim = torch.exp(sim)
        pos_sim = (exp_sim * same_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)
        
        loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8)).mean()
        
        return loss
    
    def forward(self, embeddings, labels_dict):
        """
        Args:
            embeddings: (batch_size, 128)
            labels_dict: Dict mapping axis names to label tensors
                        e.g., {'shape': tensor([0, 1, 2, ...]),
                               'color': tensor([0, 1, 0, ...])}
        
        Returns:
            total_loss: Combined SIGReg + contrastive loss
            stats: Detailed statistics
        """
        # SIGReg component (prevents collapse)
        sigreg_loss, sigreg_stats = self.sigreg(embeddings)
        
        # Contrastive component (semantic discrimination)
        contrastive_loss = 0.0
        contrastive_stats = {}
        
        for axis_name, labels in labels_dict.items():
            if axis_name in self.axis_dims:
                c_loss = self.contrastive_loss(embeddings, labels, axis_name)
                contrastive_loss += c_loss
                contrastive_stats[axis_name] = c_loss.item()
        
        # Combine
        total_loss = (
            self.lambda_sigreg * sigreg_loss +
            self.lambda_contrastive * contrastive_loss
        )
        
        stats = {
            'sigreg': sigreg_stats,
            'contrastive': contrastive_stats,
            'total': total_loss.item()
        }
        
        return total_loss, stats


# Example usage
if __name__ == "__main__":
    # CORTEX-12 axis configuration
    axis_dims = {
        'shape': (0, 31),
        'size': (32, 47),
        'material': (48, 63),
        'color': (64, 79),
        'location': (80, 87),
        'orientation': (88, 103)
    }
    
    # Create hybrid loss
    criterion = HybridSIGRegContrastive(
        axis_dims=axis_dims,
        temperature=0.1,
        lambda_sigreg=0.5,
        lambda_contrastive=0.5
    )
    
    # Dummy batch
    batch_size = 32
    embeddings = torch.randn(batch_size, 128)
    labels_dict = {
        'shape': torch.randint(0, 3, (batch_size,)),
        'color': torch.randint(0, 8, (batch_size,)),
        'size': torch.randint(0, 5, (batch_size,))
    }
    
    # Compute loss
    loss, stats = criterion(embeddings, labels_dict)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
