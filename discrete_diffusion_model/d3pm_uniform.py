import math
from typing import Optional

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, Tensor

from dit import DiT


def onehot(x: Tensor, K: int):
    return F.one_hot(x, K).float() if x.ndim == 2 else x.clone()


class D3PMUniform(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        n_blocks: int,
        n_cond: int,
        dropout: float,
        T: int,
        lambda_ce: float,
        num_samples: int = 20,  # Number of t samples to average over
    ) -> None:
        super().__init__()

        self.T = T
        self.lambda_ce = lambda_ce
        self.K = vocab_size  # No +1 since we don't have an absorbing state
        self.net = DiT(self.K, n_embed, n_heads, n_blocks, n_cond, dropout)
        self.eps = 1e-20
        self.num_samples = num_samples

        # Calculate beta schedule (1-indexed for simplicity)
        # beta(0) = undef, beta(1) = 1/T, beta(2) = 1/(T-1), ..., beta(T) = 1
        betas = torch.reciprocal(T - torch.arange(T + 1) + 1)
        betas[0] = 0.0

        # Calculate alpha and alpha_bar values
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        # Uniform transition probability (equal chance to transition to any token)
        self.uniform_prob = 1.0 / self.K

    def mul_Qbar(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute q(x_t | x_0) using uniform noise transition

        Args:
            x: class indices (BL) or probabilities (BLK) at t = 0
            t: timestep indices (B)

        Returns:
            Tensor: probability distribution over vocabulary at time t
        """
        y = onehot(x, self.K)  # [B, L, K]
        alpha_bar_t = self.alpha_bars[t][:, None, None]  # [B, 1, 1]

        # q(x_t | x_0) = α_bar_t * one_hot(x_0) + (1 - α_bar_t) * uniform
        # The probability of remaining the same token
        stay_prob = alpha_bar_t

        # The probability of changing to any other token (uniform distribution)
        change_prob = (1 - alpha_bar_t) * self.uniform_prob

        # Add uniform noise to the one-hot distribution
        return stay_prob * y + change_prob

    def mul_QT(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute q(x_t+1 | x_t) as a function of x_t+1

        Args:
            x: sample at t+1
            t: timestep indices (B)

        Returns:
            Tensor: transition probability q(x_t+1 | x_t) as function of x_t+1
        """
        y = onehot(x, self.K)
        beta_t = self.betas[t][:, None, None]  # [B, 1, 1]

        # Calculate the probability of each token at time t transitioning to each token at t+1
        # For uniform transitions, this is:
        # - If x_t+1 == x_t: probability = 1 - β_t + β_t * (1/K)
        # - If x_t+1 != x_t: probability = β_t * (1/K)

        # Create a uniform distribution tensor
        uniform_dist = torch.full_like(y, self.uniform_prob)

        # Weighted sum of identity matrix and uniform distribution
        return (1 - beta_t) * y + beta_t * uniform_dist

    def compute_unnormalized_log_posterior(self, x_0, t, x_tplus1=None) -> tuple[Tensor, Tensor]:
        """Compute log of unnormalized posterior probs q(x_t | x_t+1, x_0).

        Args:
            x_0: class indices (BL) or predicted probabilities (BLK) at t = 0
            t: transition times LongTensor with values in [0, T-1] (B)
            x_tplus1 (optional): Sample from q(x_t+1 | x_0) if already computed (BL)

        Returns:
            tuple: (unnormalized posterior log probs (BLK), x_tplus1 sample (BL))
        """
        # Compute q(x_t+1 | x_0) = q_0 @ Qbar_t+1
        q_x_tplus1_given_x_0 = self.mul_Qbar(x_0, t + 1)

        if x_tplus1 is None:
            # Sample x_t+1 from q(x_t+1 | x_0)
            x_tplus1 = D.Categorical(probs=q_x_tplus1_given_x_0).sample()

        # Multiply x_t+1 with transpose(Q_t+1) to get q(x_t+1 | x_t)
        q_x_tplus1_given_x_t = self.mul_QT(x_tplus1, t + 1)

        # Compute q(x_t | x_0) = q_0 @ Qbar_t
        q_x_t_given_x_0 = self.mul_Qbar(x_0, t)

        # Compute unnormalized posterior log probs
        # log[q(x_t | x_t+1, x_0)] ∝ log[q(x_t+1 | x_t)] + log[q(x_t | x_0)]
        log_posterior = torch.log(q_x_tplus1_given_x_t + self.eps) + torch.log(q_x_t_given_x_0 + self.eps)

        # If t = 0, simply set posterior to x_0
        if x_0.ndim == 2:
            x_0 = F.one_hot(x_0, self.K).float()
        log_posterior = torch.where(t[:, None, None] == 0, torch.log(x_0 + self.eps), log_posterior)

        return log_posterior, x_tplus1

    def forward(self, data: Tensor, t: Optional[Tensor] = None) -> tuple[Tensor, Tensor, dict]:
        """Returns the output params, training loss, and dict with useful items to log

        Averages over multiple t values for more stable training
        """
        batch_size = data.size(0)

        # Initialize accumulators for averaged results
        avg_log_predicted_x_0 = None
        avg_loss = 0.0
        avg_metrics = {
            'kl': 0.0,
            'ce': 0.0,
            'l_T': 0.0,
            'bpt': 0.0
        }

        # Sample t values if not provided
        if t is None:
            # We'll sample num_samples different t values for each item in the batch
            # For stability, we ensure at least one t=0 for reconstruction loss

            # First, ensure we have at least one t=0 for direct reconstruction
            for sample_idx in range(self.num_samples):
                # For the first sample, use explicit t values including at least one t=0
                if sample_idx == 0:
                    # Ensure at least one t=0 in the batch
                    current_t = torch.randint(0, self.T, (batch_size,), device=data.device)
                    if 0 not in current_t:
                        # Replace a random position with t=0
                        rand_idx = torch.randint(0, batch_size, (1,))
                        current_t[rand_idx] = 0
                else:
                    # For other samples, just use random t values
                    current_t = torch.randint(0, self.T, (batch_size,), device=data.device)

                # Process this batch with the current t values
                log_predicted_x_0, loss, metrics = self._forward_single(data, current_t)

                # For the first prediction, initialize the average
                if avg_log_predicted_x_0 is None:
                    avg_log_predicted_x_0 = log_predicted_x_0 / self.num_samples
                else:
                    avg_log_predicted_x_0 += log_predicted_x_0 / self.num_samples

                # Accumulate loss and metrics
                avg_loss += loss / self.num_samples
                for k, v in metrics.items():
                    avg_metrics[k] += v / self.num_samples
        else:
            # If t is explicitly provided, just use it directly (for validation/testing)
            avg_log_predicted_x_0, avg_loss, avg_metrics = self._forward_single(data, t)

        return avg_log_predicted_x_0, avg_loss, avg_metrics

    def _forward_single(self, data: Tensor, t: Tensor) -> tuple[Tensor, Tensor, dict]:
        """Helper method to compute a single forward pass with given t values"""
        # 1. Compute the log posterior: first sample from q(x_{t+1} | x_0), then compute q(x_t | x_{t+1}, x_0)
        log_q, x_tplus1 = self.compute_unnormalized_log_posterior(data, t)

        # 2. Predict x_0 and use it to compute p(x_t | x_{t+1})
        log_predicted_x_0 = self.net(x_tplus1, (t + 1).float())
        p_x_0 = F.softmax(log_predicted_x_0, dim=-1)
        log_p, _ = self.compute_unnormalized_log_posterior(p_x_0, t, x_tplus1)

        # 3. Compute KL(q || p)
        l_kl = F.softmax(log_q, dim=-1) * (F.log_softmax(log_q, dim=-1) - F.log_softmax(log_p, dim=-1))
        l_kl = F.relu(l_kl.sum(dim=-1))  # stability trick from official impl

        # 4. Compute CE for use as auxiliary loss and l_0
        l_ce = F.cross_entropy(log_predicted_x_0.view(-1, self.K), data.flatten(), reduction="none").view_as(data)

        loss = l_kl + self.lambda_ce * l_ce
        loss = torch.where(t[:, None] == 0, l_ce, loss)

        # 5. Compute an estimate of the T-step loss
        l_0 = l_ce[t == 0]
        l_kl = l_kl[t > 0]  # this is l_{T-1}
        if l_0.numel() > 0:
            l_T = l_kl.mean() * (self.T - 1) + l_0.mean()
        else:
            l_T = l_kl.mean() * self.T

        return log_predicted_x_0, loss.mean(), dict(kl=l_kl.mean() if l_kl.numel() > 0 else torch.tensor(0.0, device=loss.device),
                                                   ce=l_ce.mean(),
                                                   l_T=l_T,
                                                   bpt=l_T / math.log(2))

    def sample(self, shape: tuple, device: torch.device, x_T: Optional[Tensor] = None) -> Tensor:
        """Sample from the model using ancestral sampling

        Args:
            shape: Shape of samples to generate (batch_size, seq_len)
            device: Device to generate samples on
            x_T: Optional starting point at time T (default: sample uniformly)

        Returns:
            Tensor: Generated samples of shape (batch_size, seq_len)
        """
        batch_size, seq_len = shape

        # If no starting point provided, sample uniformly
        if x_T is None:
            x_T = torch.randint(0, self.K, (batch_size, seq_len), device=device)

        x_t = x_T

        # Ancestral sampling from t=T to t=0
        for t in range(self.T, 0, -1):
            # Get model prediction for x_0
            log_x_0 = self.net(x_t, torch.full((batch_size,), t, device=device))
            p_x_0 = F.softmax(log_x_0, dim=-1)

            # If t=1, directly output predicted x_0
            if t == 1:
                x_t = p_x_0.argmax(dim=-1) if shape[-1] > 0 else p_x_0
                break

            # Compute q(x_{t-1} | x_t, x_0) using Bayes rule
            # log_posterior[b, i, k] = log p(x_{t-1}^i = k | x_t^i, x_0^i)
            log_posterior, _ = self.compute_unnormalized_log_posterior(
                p_x_0,
                torch.full((batch_size,), t-1, device=device),
                x_t
            )

            posterior = F.softmax(log_posterior, dim=-1)

            # Sample x_{t-1} from the posterior
            x_t = D.Categorical(probs=posterior).sample()

        return x_t
