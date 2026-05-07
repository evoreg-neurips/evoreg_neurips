"""
Diffusion process manager for iterative point cloud refinement.

Handles the forward diffusion process (adding noise) and reverse
denoising process for training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class NoiseSchedule:
    """
    Manages the noise schedule for diffusion process.
    
    Defines how much noise to add at each timestep and
    provides utilities for the diffusion process.
    """
    
    def __init__(
        self,
        n_timesteps: int = 1000,
        schedule_type: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Initializes noise schedule.
        
        Args:
            n_timesteps: Number of diffusion timesteps
            schedule_type: Type of schedule ('linear' or 'cosine')
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.n_timesteps = n_timesteps
        self.schedule_type = schedule_type
        
        # Creates beta schedule (variance of noise added at each step)
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif schedule_type == 'cosine':
            # Cosine schedule for more stable training
            steps = torch.arange(n_timesteps + 1) / n_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            self.betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Precomputes useful quantities
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
    
    def get_variance(self, t: int) -> float:
        """Gets variance for timestep t."""
        return self.betas[t].item()
    
    def get_alpha_bar(self, t: int) -> float:
        """Gets cumulative product of alphas up to timestep t."""
        return self.alphas_cumprod[t].item()


class DiffusionProcess(nn.Module):
    """
    Manages the complete diffusion process for point cloud refinement.
    
    Handles both forward process (adding noise during training) and
    reverse process (denoising during inference).
    """
    
    def __init__(
        self,
        score_network: nn.Module,
        n_timesteps: int = 1000,
        schedule_type: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Initializes diffusion process.
        
        Args:
            score_network: Network that predicts denoising score
            n_timesteps: Number of diffusion timesteps
            schedule_type: Type of noise schedule
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        super(DiffusionProcess, self).__init__()
        
        self.score_network = score_network
        self.n_timesteps = n_timesteps
        
        # Initializes noise schedule
        self.noise_schedule = NoiseSchedule(
            n_timesteps=n_timesteps,
            schedule_type=schedule_type,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        # Registers buffers for efficient computation
        self.register_buffer('betas', self.noise_schedule.betas)
        self.register_buffer('alphas', self.noise_schedule.alphas)
        self.register_buffer('alphas_cumprod', self.noise_schedule.alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', self.noise_schedule.sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           self.noise_schedule.sqrt_one_minus_alphas_cumprod)
    
    def forward_diffusion(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process - adds noise to clean data.
        
        This is used during training to create noisy samples.
        
        Args:
            x_0: Clean point cloud (B, N, 3)
            t: Timesteps for each sample (B,)
            noise: Optional pre-generated noise
        
        Returns:
            Tuple of (noisy points, noise used)
        """
        batch_size = x_0.shape[0]
        
        # Generates noise if not provided
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Gets noise parameters for each sample's timestep
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(batch_size, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1)
        
        # Adds noise according to schedule
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        target: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes diffusion training loss.
        
        Args:
            x_0: Clean point cloud from VAE (B, N, 3)
            target: Target point cloud (B, M, 3)
            t: Optional timesteps (if None, samples randomly)
        
        Returns:
            Dictionary with loss values
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Samples random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=device)
        
        # Forward diffusion - adds noise
        noise = torch.randn_like(x_0)
        x_t, true_noise = self.forward_diffusion(x_0, t, noise)
        
        # Predicts noise using score network
        predicted_score = self.score_network(x_t, target, t)
        
        # Score matching loss
        # The score is related to the noise by: score = -noise / sigma
        # So we train to predict the noise directly
        loss = F.mse_loss(predicted_score, -true_noise)
        
        return {
            'diffusion_loss': loss,
            'noise_pred_error': (predicted_score + true_noise).abs().mean()
        }
    
    @torch.no_grad()
    def reverse_diffusion(
        self,
        x_t: torch.Tensor,
        target: torch.Tensor,
        start_timestep: Optional[int] = None,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Reverse diffusion process - iteratively denoises.
        
        This is used during inference to refine the VAE output.
        
        Args:
            x_t: Starting noisy points (B, N, 3)
            target: Target point cloud (B, M, 3)
            start_timestep: Starting timestep (default: n_timesteps-1)
            num_steps: Number of denoising steps (default: all)
            return_trajectory: Whether to return all intermediate states
        
        Returns:
            Denoised point cloud (or trajectory if requested)
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        if start_timestep is None:
            start_timestep = self.n_timesteps - 1
        
        if num_steps is None:
            num_steps = start_timestep + 1
        
        # Initializes trajectory storage
        if return_trajectory:
            trajectory = [x_t]
        
        # Current state
        x = x_t
        
        # Reverse diffusion loop
        timesteps = torch.linspace(start_timestep, 0, num_steps, dtype=torch.long, device=device)
        
        for t_idx in timesteps:
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            
            # Predicts score
            score = self.score_network(x, target, t)
            
            # Gets noise parameters
            beta_t = self.betas[t_idx]
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t_idx]
            sqrt_recip_alpha = 1.0 / torch.sqrt(self.alphas[t_idx])
            
            # Denoising step (DDPM update rule)
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t + beta_t * score) + sigma_t * z
            model_mean = sqrt_recip_alpha * (x + beta_t * score / sqrt_one_minus_alpha_bar)
            
            # Adds noise for all steps except the last
            if t_idx > 0:
                posterior_variance = self.noise_schedule.posterior_variance[t_idx]
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = model_mean
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return torch.stack(trajectory, dim=0)  # (T, B, N, 3)
        else:
            return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        x_t: torch.Tensor,
        target: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM sampling - deterministic and faster denoising.
        
        Args:
            x_t: Starting noisy points (B, N, 3)
            target: Target point cloud (B, M, 3)
            num_steps: Number of denoising steps (fewer than training)
            eta: Stochasticity parameter (0 = deterministic)
        
        Returns:
            Denoised point cloud
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Creates subset of timesteps
        step_size = self.n_timesteps // num_steps
        timesteps = torch.arange(0, self.n_timesteps, step_size, device=device).flip(0)
        
        x = x_t
        
        for i in range(len(timesteps) - 1):
            t = torch.full((batch_size,), timesteps[i], device=device)
            t_next = torch.full((batch_size,), timesteps[i + 1], device=device)
            
            # Predicts score
            score = self.score_network(x, target, t)
            
            # DDIM update rule
            alpha_t = self.alphas_cumprod[timesteps[i]]
            alpha_t_next = self.alphas_cumprod[timesteps[i + 1]]
            
            # Predicts x_0
            x_0_pred = (x + score * torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_x_t = torch.sqrt(1 - alpha_t_next - eta**2) * score
            
            # Random noise
            noise = eta * torch.randn_like(x) if eta > 0 else 0
            
            # Next step
            x = torch.sqrt(alpha_t_next) * x_0_pred + dir_x_t + noise
        
        return x
    
    def refine(
        self,
        vae_output: torch.Tensor,
        target: torch.Tensor,
        num_steps: int = 50,
        noise_level: float = 0.1,
        use_ddim: bool = False
    ) -> torch.Tensor:
        """
        Refines VAE output through diffusion process.
        
        This is the main interface for inference - takes VAE output
        and refines it through iterative denoising.
        
        Args:
            vae_output: Initial registration from VAE (B, N, 3)
            target: Target point cloud (B, M, 3)
            num_steps: Number of refinement steps
            noise_level: Initial noise to add (0-1, fraction of schedule)
            use_ddim: Whether to use DDIM sampling (faster)
        
        Returns:
            Refined point cloud
        """
        # Adds small amount of noise to VAE output
        # This treats it as a slightly noisy version that needs refinement
        start_timestep = int(noise_level * self.n_timesteps)
        
        if start_timestep > 0:
            t = torch.full((vae_output.shape[0],), start_timestep, device=vae_output.device)
            x_t, _ = self.forward_diffusion(vae_output, t)
        else:
            x_t = vae_output
        
        # Refines through reverse diffusion
        if use_ddim:
            refined = self.ddim_sample(x_t, target, num_steps)
        else:
            refined = self.reverse_diffusion(x_t, target, start_timestep, num_steps)
        
        return refined


class SimplifiedDiffusion(nn.Module):
    """
    Simplified diffusion for easier integration and testing.
    """
    
    def __init__(self, score_network: nn.Module, n_timesteps: int = 100):
        """
        Initializes simplified diffusion.
        
        Args:
            score_network: Score prediction network
            n_timesteps: Number of timesteps
        """
        super(SimplifiedDiffusion, self).__init__()
        
        self.score_network = score_network
        self.n_timesteps = n_timesteps
        
        # Simple linear schedule
        self.betas = torch.linspace(0.0001, 0.02, n_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x_0: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        """Adds noise for timestep t."""
        alpha_bar = self.alphas_bar[t]
        return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    
    def denoise_step(
        self,
        x_t: torch.Tensor,
        target: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """Single denoising step."""
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device)
        
        # Predicts score
        score = self.score_network(x_t, target, t_tensor)
        
        # Simple denoising
        alpha = self.alphas[t]
        beta = self.betas[t]
        
        # Mean of posterior
        mean = (1 / torch.sqrt(alpha)) * (x_t + beta * score / torch.sqrt(1 - self.alphas_bar[t]))
        
        # Adds noise except for last step
        if t > 0:
            noise = torch.randn_like(x_t) * torch.sqrt(beta)
            return mean + noise
        else:
            return mean


if __name__ == "__main__":
    # Tests diffusion process
    print("Testing Diffusion Process...")
    
    # Creates dummy score network
    from score_network import ScoreNetwork
    
    score_net = ScoreNetwork(
        point_dim=3,
        hidden_dim=128,
        time_embed_dim=64,
        n_layers=2
    )
    
    # Creates diffusion process
    diffusion = DiffusionProcess(
        score_network=score_net,
        n_timesteps=100,
        schedule_type='linear'
    )
    
    # Test data
    batch_size = 2
    n_points = 500
    vae_output = torch.randn(batch_size, n_points, 3)
    target = torch.randn(batch_size, n_points, 3)
    
    # Tests forward diffusion
    print("\n1. Testing forward diffusion:")
    t = torch.tensor([10, 50])
    noisy, noise = diffusion.forward_diffusion(vae_output, t)
    print(f"Original shape: {vae_output.shape}")
    print(f"Noisy shape: {noisy.shape}")
    print(f"Noise level at t=10: {diffusion.noise_schedule.get_variance(10):.4f}")
    print(f"Noise level at t=50: {diffusion.noise_schedule.get_variance(50):.4f}")
    
    # Tests loss computation
    print("\n2. Testing loss computation:")
    losses = diffusion.compute_loss(vae_output, target)
    print(f"Diffusion loss: {losses['diffusion_loss'].item():.4f}")
    print(f"Noise prediction error: {losses['noise_pred_error'].item():.4f}")
    
    # Tests reverse diffusion
    print("\n3. Testing reverse diffusion:")
    refined = diffusion.reverse_diffusion(noisy, target, start_timestep=50, num_steps=10)
    print(f"Refined shape: {refined.shape}")
    improvement = (refined - noisy).norm(dim=-1).mean()
    print(f"Average refinement: {improvement.item():.4f}")
    
    # Tests DDIM sampling
    print("\n4. Testing DDIM sampling:")
    ddim_refined = diffusion.ddim_sample(noisy, target, num_steps=10)
    print(f"DDIM refined shape: {ddim_refined.shape}")
    
    # Tests main refinement interface
    print("\n5. Testing refinement interface:")
    final_refined = diffusion.refine(
        vae_output,
        target,
        num_steps=20,
        noise_level=0.1,
        use_ddim=True
    )
    print(f"Final refined shape: {final_refined.shape}")
    
    # Tests simplified version
    print("\n6. Testing simplified diffusion:")
    simple_diff = SimplifiedDiffusion(score_net, n_timesteps=50)
    noisy_simple = simple_diff.add_noise(vae_output, t=25, noise=torch.randn_like(vae_output))
    denoised_simple = simple_diff.denoise_step(noisy_simple, target, t=25)
    print(f"Simplified denoised shape: {denoised_simple.shape}")
    
    print("\nAll Diffusion tests passed!")