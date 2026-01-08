import torch
import torch.nn as nn


class WindowNN(nn.Module):
    """
    NN surrogate for one HR output from one time window of IBI.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        """
        Args:
            input_dim: fixed window length L
            hidden_dim: small number (e.g. 16)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_window):
        """
        Args:
            x_window: Tensor of shape (L,) or (batch_size, L)
        
        Returns:
            Tensor of shape (1,) or (batch_size, 1)
        """
        return self.net(x_window)

