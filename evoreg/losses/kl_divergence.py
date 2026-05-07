"""
KL divergence loss for VAE training.

Implements the Kullback-Leibler divergence between the learned latent
distribution and a prior distribution (typically standard Gaussian).
"""

import torch
import torch.nn as nn
from typing import Optional


def kl_divergence(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes KL divergence between a Gaussian and standard normal.
    
    Calculates KL(N(mu, sigma^2) || N(0, I)) where sigma^2 = exp(log_var).
    This is the standard KL term used in VAE training.
    
    Args:
        mu: Mean of the learned distribution (B, D) or (D,)
        log_var: Log variance of the learned distribution (B, D) or (D,)
        reduction: Reduction method ('mean', 'sum', 'batchmean', or 'none')
        
    Returns:
        KL divergence value(s)
    """
    # Computes KL divergence using the closed-form solution
    # KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    
    # Applies reduction
    if reduction == 'mean':
        # Averages over all dimensions and batch
        return kl.mean()
    elif reduction == 'sum':
        # Sums over all dimensions and batch
        return kl.sum()
    elif reduction == 'batchmean':
        # Sums over dimensions, averages over batch
        return kl.sum(dim=-1).mean()
    elif reduction == 'none':
        # Returns per-sample KL divergence
        return kl.sum(dim=-1)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss module for VAE training.
    
    Provides a PyTorch module wrapper for computing KL divergence
    with optional annealing and clamping for stable training.
    """
    
    def __init__(
        self,
        reduction: str = 'batchmean',
        beta: float = 1.0,
        free_bits: Optional[float] = None,
        max_capacity: Optional[float] = None
    ):
        """
        Initializes the KL divergence loss module.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'batchmean', or 'none')
            beta: Weight for the KL term (beta-VAE)
            free_bits: Minimum KL value per dimension (free bits technique)
            max_capacity: Maximum KL capacity for gradual increase
        """
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction
        self.beta = beta
        self.free_bits = free_bits
        self.max_capacity = max_capacity
        
        # Tracks the current capacity for annealing
        self.current_capacity = 0.0 if max_capacity is not None else None
    
    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Computes the KL divergence loss.
        
        Args:
            mu: Mean of the learned distribution (B, D)
            log_var: Log variance of the learned distribution (B, D)
            step: Current training step for capacity annealing
            
        Returns:
            Weighted KL divergence loss
        """
        # Computes per-dimension KL divergence
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        
        # Applies free bits if specified
        if self.free_bits is not None:
            # Clamps each dimension to have at least free_bits nats
            kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(self.free_bits))
        
        # Sums over latent dimensions
        kl_per_sample = kl_per_dim.sum(dim=-1)
        
        # Applies capacity constraint if specified
        if self.max_capacity is not None:
            # Updates current capacity based on training step
            if step is not None:
                # Linear annealing over first 10000 steps
                self.current_capacity = min(self.max_capacity, 
                                           self.max_capacity * step / 10000)
            
            # Applies capacity constraint
            kl_per_sample = torch.abs(kl_per_sample - self.current_capacity)
        
        # Applies reduction
        if self.reduction == 'mean':
            kl_loss = kl_per_sample.mean()
        elif self.reduction == 'sum':
            kl_loss = kl_per_sample.sum()
        elif self.reduction == 'batchmean':
            kl_loss = kl_per_sample.mean()
        elif self.reduction == 'none':
            kl_loss = kl_per_sample
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")
        
        # Applies beta weighting
        kl_loss = self.beta * kl_loss
        
        return kl_loss
    
    def update_beta(self, new_beta: float):
        """
        Updates the beta weight for KL annealing.
        
        Args:
            new_beta: New beta value
        """
        self.beta = new_beta
    
    def extra_repr(self) -> str:
        """
        Returns extra representation string for printing.
        
        Returns:
            String with module parameters
        """
        repr_str = f'reduction={self.reduction}, beta={self.beta}'
        if self.free_bits is not None:
            repr_str += f', free_bits={self.free_bits}'
        if self.max_capacity is not None:
            repr_str += f', max_capacity={self.max_capacity}'
        return repr_str


def kl_divergence_normal(
    mu1: torch.Tensor,
    log_var1: torch.Tensor,
    mu2: torch.Tensor,
    log_var2: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes KL divergence between two Gaussian distributions.
    
    Calculates KL(N(mu1, sigma1^2) || N(mu2, sigma2^2)).
    
    Args:
        mu1: Mean of the first distribution (B, D)
        log_var1: Log variance of the first distribution (B, D)
        mu2: Mean of the second distribution (B, D)
        log_var2: Log variance of the second distribution (B, D)
        reduction: Reduction method ('mean', 'sum', or 'none')
        
    Returns:
        KL divergence value(s)
    """
    # Computes KL divergence between two Gaussians
    # KL = 0.5 * (log(sigma2^2/sigma1^2) + (sigma1^2 + (mu1-mu2)^2)/sigma2^2 - 1)
    var1 = log_var1.exp()
    var2 = log_var2.exp()
    
    kl = 0.5 * (log_var2 - log_var1 + var1/var2 + (mu1 - mu2).pow(2)/var2 - 1)
    
    # Applies reduction
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    elif reduction == 'none':
        return kl.sum(dim=-1)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


if __name__ == "__main__":
    # Tests the KL divergence implementation
    print("Testing KL divergence implementation...")
    
    # Creates test latent distributions
    batch_size = 4
    latent_dim = 128
    
    mu = torch.randn(batch_size, latent_dim) * 0.1
    log_var = torch.randn(batch_size, latent_dim) * 0.1 - 2.0  # Small variance
    
    # Tests functional interface
    kl_loss = kl_divergence(mu, log_var, reduction='batchmean')
    print(f"KL divergence (functional): {kl_loss.item():.4f}")
    
    # Tests module interface
    kl_module = KLDivergenceLoss(reduction='batchmean', beta=1.0)
    loss = kl_module(mu, log_var)
    print(f"KL divergence (module): {loss.item():.4f}")
    
    # Tests with beta weighting
    kl_beta = KLDivergenceLoss(reduction='batchmean', beta=0.5)
    loss_beta = kl_beta(mu, log_var)
    print(f"KL divergence (beta=0.5): {loss_beta.item():.4f}")
    
    # Tests with free bits
    kl_free = KLDivergenceLoss(reduction='batchmean', free_bits=0.01)
    loss_free = kl_free(mu, log_var)
    print(f"KL divergence (free bits): {loss_free.item():.4f}")
    
    # Tests gradient flow
    mu_grad = torch.randn(batch_size, latent_dim, requires_grad=True)
    log_var_grad = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    loss_grad = kl_module(mu_grad, log_var_grad)
    loss_grad.backward()
    
    print(f"\nGradient shapes:")
    print(f"  mu gradient: {mu_grad.grad.shape}")
    print(f"  log_var gradient: {log_var_grad.grad.shape}")
    
    # Tests KL between two Gaussians
    mu2 = torch.randn(batch_size, latent_dim) * 0.1
    log_var2 = torch.randn(batch_size, latent_dim) * 0.1 - 2.0
    
    kl_two = kl_divergence_normal(mu, log_var, mu2, log_var2)
    print(f"\nKL between two Gaussians: {kl_two.item():.4f}")
    
    print("\nAll KL divergence tests passed!")