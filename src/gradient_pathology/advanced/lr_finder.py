"""Learning rate range test for optimal LR discovery."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Tuple, List
from copy import deepcopy


class LRFinder:
    """Automatically find optimal learning rate using range test.
    
    Based on Leslie Smith's LR Range Test:
    - Start with very small LR
    - Exponentially increase LR
    - Track loss at each LR
    - Optimal LR is where loss decreases fastest
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = "cpu"):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history: List[Tuple[float, float]] = []

    def range_test(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
    ) -> Tuple[List[float], List[float]]:
        """Run LR range test.
        
        Args:
            dataloader: Training data
            loss_fn: Loss function
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            
        Returns:
            (learning_rates, losses) lists
        """
        # Save original state
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
        losses = []
        
        self.model.train()
        data_iter = iter(dataloader)
        
        for lr in lrs:
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data, target = next(data_iter)
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            self.history.append((lr, loss.item()))
            
            # Stop if loss explodes
            if len(losses) > 1 and losses[-1] > 4 * min(losses):
                break
        
        # Restore original state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        return list(lrs[:len(losses)]), losses
    
    def suggest_lr(self, lrs: List[float], losses: List[float]) -> float:
        """Suggest optimal learning rate.
        
        Finds LR where loss decreases most rapidly.
        """
        # Smooth losses
        losses_array = np.array(losses)
        
        # Find steepest gradient (most negative)
        gradients = np.gradient(losses_array)
        min_grad_idx = np.argmin(gradients)
        
        # Suggested LR is slightly before steepest point
        suggested_idx = max(0, min_grad_idx - len(losses) // 10)
        
        return lrs[suggested_idx]
    
    def plot(self, lrs: List[float], losses: List[float], save_path: str = None) -> None:
        """Plot LR range test results."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(lrs, losses, linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Range Test')
        ax.grid(True, alpha=0.3)
        
        # Mark suggested LR
        suggested_lr = self.suggest_lr(lrs, losses)
        suggested_loss = losses[lrs.index(suggested_lr)]
        ax.axvline(suggested_lr, color='r', linestyle='--', 
                   label=f'Suggested LR: {suggested_lr:.2e}')
        ax.plot(suggested_lr, suggested_loss, 'ro', markersize=10)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
