import numpy as np
import torch
from torch import nn

# -------------------- Unit Dot Product Similarity (between 2 single vector) --------------------
# --- numpy version ---
a = np.array([1.0, 2.0])
b = np.array([2.0, 4.0])
norm_a = np.linalg.norm(a, axis=-1)
norm_b = np.linalg.norm(b, axis=-1)
if norm_a + norm_b > 0:
    s = 4 * np.dot(a, b) / (norm_a + norm_b) ** 2
else:
    s = 0
print(s, "(numpy version)")
# --- torch version ---
tensor_a = torch.from_numpy(a)
tensor_b = torch.from_numpy(b)
norm_a = torch.norm(tensor_a, dim=-1, p=2)
norm_b = torch.norm(tensor_b, dim=-1, p=2)
if norm_a + norm_b > 0:
    s = (4 * a @ b) / (norm_a + norm_b) ** 2
else:
    s = 0
print(s, "(torch version)")

# -------------------- Unit Dot Product Similarity (between 2 vectors group) --------------------
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 5.0], [-3.0, -1.0, -2.0], [2.0, -1.0, -2.0]]
B = [[3.0, -1.0, -2.0], [-1.0, -2.0, -3.0], [3.0, 5.0, 5.0], [2.0, -1.0, -2.0]]
# (B,N,D)
A = torch.tensor(A).unsqueeze(0)
B = torch.tensor(B).unsqueeze(0)
# (B,N)
norm_A = torch.norm(A, dim=-1, p=2)
norm_B = torch.norm(B, dim=-1, p=2)
# (N,N)
D = (norm_A.unsqueeze(-1) + norm_B.unsqueeze(-2)) ** 2
S_u_dot = 4 * A @ B.transpose(-1, -2) / D.masked_fill(D.eq(0.0), float('inf'))
print(S_u_dot, "(similarities between 2 vectors group)")


# ------ a code implementation example of Multi-Head Self-Attention with Unit Dot Product Similarity ------
class SimiMultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, model_dim: int):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert model_dim % n_heads == 0
        self.d_k = model_dim // n_heads
        self.n_heads = n_heads
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, model_dim)
        self.alpha = torch.nn.Parameter(torch.ones(1, n_heads, 1, 1) * 10.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q,k,v (B,T,D) B is batchsize, T is number of time steps, D is model_dim
        B, T, D = q.shape
        # (B,T,D) -> (B,T,n_heads,d_k) -> (B,n_heads,T,d_k)
        q = self.linear_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # dot product {(B,n_heads,T,d_k) @ (B,n_heads,T,d_k) -> (B,n_heads,T,T)}
        scores = q @ k.transpose(-2, -1)
        # compute norm of q, k {(B,n_heads,T,d_k) -> (B,n_heads,T)}
        norm_q = torch.norm(q, p=2, dim=-1)
        norm_k = torch.norm(k, p=2, dim=-1)
        # compute square of the sum of norms {(B,n_heads,T,1) + (B,n_heads,1,T) -> (B,n_heads,T,T)}
        D_udps = (norm_q.unsqueeze(-1) + norm_k.unsqueeze(-2)) ** 2
        # divide by square of the sum of norms {(B,n_heads,T,T)}
        scores = self.alpha * 4 * scores / D_udps.masked_fill(D_udps.eq(0.0), float('inf'))
        
        # do softmax {(B,n_heads,T,T)}
        scores = torch.softmax(scores, dim=-1)
        # weighted sum {(B,n_heads,T,T) @ (B,n_heads,T,d_k) -> (B,n_heads,T,d_k)}
        output = scores @ v
        # output linear {(B,n_heads,T,d_k) -> (B,T,n_heads,d_k) -> (B,T,D) -> (B,T,D) D=n_heads*d_k}
        output = self.linear_out(output.transpose(1, 2).reshape(B, T, -1))

        # (B,T,D)
        return output
