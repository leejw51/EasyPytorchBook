# Easy Transformers v1.0

**Neural Network Mathematics & Transformer Architecture**

*by jw*

This book covers transformer architecture and neural network mathematics:

- **Attention Mechanisms** - Self-attention, multi-head attention
- **Backpropagation** - Gradient flow through transformers
- **Training Dynamics** - Optimization and regularization
- **Code Examples** - PyTorch implementations

*Using PyTorch 2.8.0*

---

## Table of Contents

1. [Transformer Architecture Flow](#transformer-architecture-flow)
2. [Self-Attention: The Core](#self-attention-the-core)
3. [Softmax + CrossEntropy Magic](#softmax--crossentropy-magic)
4. [Activations: Non-linearity Power](#activations-non-linearity-power)
5. [Feed Forward Network (FFN)](#feed-forward-network-(ffn))
6. [Layer Normalization Flow](#layer-normalization-flow)
7. [Positional Encoding](#positional-encoding)
8. [Gradient Flow & Backprop](#gradient-flow--backprop)
9. [Training Dynamics](#training-dynamics)
10. [Attention Patterns](#attention-patterns)
11. [Embeddings & Representations](#embeddings--representations)
12. [Masked Language Modeling](#masked-language-modeling)
13. [Autoregressive Generation](#autoregressive-generation)
14. [Scale & Emergence](#scale--emergence)
15. [Optimization Tricks](#optimization-tricks)
16. [Modern Improvements](#modern-improvements)

---

## 🔄 Transformer Architecture Flow

- Input Flow: Text -> Tokens -> Embeddings + Position
- 1. Tokenization: 'Hello' -> [H, e, l, l, o] -> [15496, 8894, 75, 75, 78]
- 2. Embedding: token_id -> E[token_id] in R^d
- 3. Position: PE(pos,2i) = sin(pos/10000^(2i/d))
- 4. Input: x = Embedding + PositionalEncoding
- 5. N x TransformerBlock: x -> MultiHeadAttn -> FFN
- 6. Output: Final layer -> Softmax -> Probabilities
- Flow: Input -> Embed -> N x [Attn+FFN] -> Output
- Each block: x -> Attn(x) + x -> FFN(x) + x
- Residual connections: prevent vanishing gradients
- Layer norm: stabilize training at each step

### Code Examples

#### Transformer Block

*A transformer block combines self-attention, feed-forward network, layer normalization, and residual connections.*

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    '''Single transformer block: Attention + FFN with residuals and LayerNorm'''
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ln2(x)

        return x

# Create and test
block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
x = torch.randn(2, 10, 64)  # batch=2, seq=10, d_model=64

output = block(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")

n_params = sum(p.numel() for p in block.parameters())
print(f"\nParameters: {n_params:,}")
print("\nBlock = Attention + LayerNorm + FFN + LayerNorm + Residuals")
```

**Output:**

```
Input shape:  torch.Size([2, 10, 64])
Output shape: torch.Size([2, 10, 64])

Parameters: 49,984

Block = Attention + LayerNorm + FFN + LayerNorm + Residuals
```

---

## 🎯 Self-Attention: The Core

- Magic Formula: Attn(Q,K,V) = softmax(QK^T/sqrt(d_k))V
- Step 1: Q=XW_Q, K=XW_K, V=XW_V (linear proj)
- Step 2: Scores = QK^T/sqrt(d_k) (scaled dot-product)
- Step 3: Weights = softmax(Scores) (attention map)
- Step 4: Output = Weights x V (weighted values)
- Why sqrt(d_k)?: Prevents softmax saturation
- Attention weights: sum_j alpha_ij = 1 (prob distribution)
- Multi-head: h heads in parallel, then concat
- MHA = Concat(head_1,...,head_h)W_O
- Each head learns different relationships
- Causal mask: prevent looking at future tokens

### Code Examples

#### Self Attention

*Self-attention computes weighted combinations of values based on query-key similarity.*

```python
import torch
import torch.nn.functional as F
import math

# Self-Attention: Attn(Q,K,V) = softmax(QK^T/sqrt(d_k))V
batch_size, seq_len, d_model = 2, 4, 8
d_k = d_model

# Input sequence
X = torch.randn(batch_size, seq_len, d_model)

# Linear projections (simplified - same weights for demo)
W_Q = torch.randn(d_model, d_k)
W_K = torch.randn(d_model, d_k)
W_V = torch.randn(d_model, d_k)

# Step 1: Compute Q, K, V
Q = X @ W_Q  # (batch, seq, d_k)
K = X @ W_K
V = X @ W_V

print(f"X shape: {X.shape}")
print(f"Q, K, V shapes: {Q.shape}")

# Step 2: Scaled dot-product attention scores
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
print(f"\nScores shape: {scores.shape}")
print(f"Scores[0]:\n{scores[0]}")

# Step 3: Softmax to get attention weights
attn_weights = F.softmax(scores, dim=-1)
print(f"\nAttention weights (sum to 1 per row):")
print(f"{attn_weights[0]}")
print(f"Row sums: {attn_weights[0].sum(dim=-1)}")

# Step 4: Weighted sum of values
output = attn_weights @ V
print(f"\nOutput shape: {output.shape}")
```

**Output:**

```
X shape: torch.Size([2, 4, 8])
Q, K, V shapes: torch.Size([2, 4, 8])

Scores shape: torch.Size([2, 4, 4])
Scores[0]:
tensor([[  2.5395,  -3.4004,   6.9126,  -3.9048],
        [-30.8605, -12.8885, -12.9827,   7.6144],
        [-21.9970,  -0.1657,  -8.6070,  -1.6373],
        [ -0.2251,   3.2522,  -6.6831,  -6.3684]])

Attention weights (sum to 1 per row):
tensor([[1.2454e-02, 3.2782e-05, 9.8749e-01, 1.9798e-05],
        [1.9524e-17, 1.2465e-09, 1.1345e-09, 1.0000e+00],
        [2.6852e-10, 8.1315e-01, 1.7546e-04, 1.8667e-01],
        [2.9962e-02, 9.6993e-01, 4.6977e-05, 6.4351e-05]])
Row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000])

Output shape: torch.Size([2, 4, 8])
```

#### Multi Head Attention

*Multi-head attention runs multiple attention operations in parallel, each learning different patterns.*

```python
import torch
import torch.nn as nn

# Multi-Head Attention
class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections and split into heads
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Combine heads
        out = (attn @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(out)

# Test
mha = SimpleMultiHeadAttention(d_model=64, num_heads=4)
x = torch.randn(2, 8, 64)  # batch=2, seq=8, d_model=64
out = mha(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Number of parameters: {sum(p.numel() for p in mha.parameters())}") 
```

**Output:**

```
Input shape: torch.Size([2, 8, 64])
Output shape: torch.Size([2, 8, 64])
Number of parameters: 16640
```

---

## 📊 Softmax + CrossEntropy Magic

- Why this combo is brilliant for gradients:
- Softmax: p_i = e^(z_i)/sum_j e^(z_j) (converts to probs)
- CrossEntropy: L = -sum_i y_i log(p_i) (measures error)
- Combined gradient: dL/dz_i = p_i - y_i
- This is incredibly simple! No complex derivatives
- Forward: z -> softmax -> p -> CE -> loss
- Backward: dL/dz = p - y (one step!)
- Softmax derivative: dp_i/dz_j = p_i(delta_ij - p_j)
- CE derivative: dL/dp_i = -y_i/p_i
- Combined: cancellation makes gradient clean
- This is why transformers train so well!

### Code Examples

#### Softmax Crossentropy

*Softmax + CrossEntropy has a beautifully simple gradient: just subtract the target from the probabilities.*

```python
import torch
import torch.nn.functional as F

# The magic of Softmax + CrossEntropy gradient
# Combined gradient: dL/dz = p - y (incredibly simple!)

# Logits (raw model output)
logits = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
# True label (one-hot)
target = torch.tensor([0])  # Class 0

print("=== Forward Pass ===")
# Softmax: convert to probabilities
probs = F.softmax(logits, dim=-1)
print(f"Logits: {logits.detach()}")
print(f"Softmax probs: {probs.detach()}")
print(f"Sum of probs: {probs.sum().item():.4f}")

# CrossEntropy loss
loss = F.cross_entropy(logits, target)
print(f"\nCrossEntropy Loss: {loss.item():.4f}")

print("\n=== Backward Pass ===")
loss.backward()

# The gradient is simply: p - y
# For class 0 (target): p[0] - 1
# For others: p[i] - 0
print(f"Gradient dL/dz: {logits.grad}")

# Verify: gradient = softmax_output - one_hot_target
one_hot = F.one_hot(target, num_classes=3).float()
expected_grad = probs.detach() - one_hot
print(f"Expected (p - y): {expected_grad}")
print(f"\nGradient is simply (p - y)! No complex math needed.")
```

**Output:**

```
=== Forward Pass ===
Logits: tensor([[2.0000, 1.0000, 0.1000]])
Softmax probs: tensor([[0.6590, 0.2424, 0.0986]])
Sum of probs: 1.0000

CrossEntropy Loss: 0.4170

=== Backward Pass ===
Gradient dL/dz: tensor([[-0.3410,  0.2424,  0.0986]])
Expected (p - y): tensor([[-0.3410,  0.2424,  0.0986]])

Gradient is simply (p - y)! No complex math needed.
```

---

## ⚡ Activations: Non-linearity Power

- Why non-linear? Linear layers can't learn complexity
- ReLU: f(x) = max(0,x), f'(x) = 1 if x>0 else 0
- GELU: x * Phi(x) (smooth, used in transformers)
- Swish: x * sigmoid(x) (smooth activation)
- Non-linearity enables: universal approximation
- Without activation: W3*W2*W1*x = W_combined*x
- With activation: can approximate any function
- Gradient flow: activation must not kill gradients
- ReLU problem: dead neurons (gradient = 0)
- GELU solution: always some gradient
- Location: typically in FFN layers

### Code Examples

#### Activations

*Modern activations like GELU and Swish provide smooth gradients, avoiding dead neuron problems.*

```python
import torch
import torch.nn.functional as F

x = torch.linspace(-3, 3, 7)
print(f"Input x: {x.tolist()}")
print()

# ReLU: max(0, x)
relu = F.relu(x)
print(f"ReLU:    {relu.tolist()}")

# GELU: x * Phi(x) - used in transformers
gelu = F.gelu(x)
print(f"GELU:    {[f'{v:.3f}' for v in gelu.tolist()]}")

# Swish/SiLU: x * sigmoid(x)
swish = F.silu(x)
print(f"Swish:   {[f'{v:.3f}' for v in swish.tolist()]}")

# Compare gradients at x=0
x_grad = torch.tensor([0.0], requires_grad=True)
F.relu(x_grad).backward()
print(f"\nGradient at x=0:")
print(f"  ReLU:  {x_grad.grad.item()} (undefined, PyTorch uses 0)")

x_grad = torch.tensor([0.0], requires_grad=True)
F.gelu(x_grad).backward()
print(f"  GELU:  {x_grad.grad.item():.4f} (smooth, non-zero)")

x_grad = torch.tensor([0.0], requires_grad=True)
F.silu(x_grad).backward()
print(f"  Swish: {x_grad.grad.item():.4f} (smooth, non-zero)")
```

**Output:**

```
Input x: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

ReLU:    [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
GELU:    ['-0.004', '-0.046', '-0.159', '0.000', '0.841', '1.954', '2.996']
Swish:   ['-0.142', '-0.238', '-0.269', '0.000', '0.731', '1.762', '2.858']

Gradient at x=0:
  ReLU:  0.0 (undefined, PyTorch uses 0)
  GELU:  0.5000 (smooth, non-zero)
  Swish: 0.5000 (smooth, non-zero)
```

---

## 🧠 Feed Forward Network (FFN)

- Structure: Linear -> Activation -> Linear
- FFN(x) = W2 * activation(W1*x + b1) + b2
- Typical: 4x expansion, then back to d_model
- Example: 768 -> 3072 -> 768 (BERT)
- Purpose: apply non-linear transformation
- Each position processed independently
- Adds model capacity and non-linearity
- Gradient: chain rule through activation
- dL/dW1 = (dL/dh) * activation'(z1) * x^T
- dL/dW2 = dL/dy * h^T
- Most parameters are in FFN layers!

### Code Examples

#### Ffn

*The FFN applies a non-linear transformation at each position, typically expanding then contracting the dimension.*

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feed Forward Network: Linear -> Activation -> Linear
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x):
        # FFN(x) = W2 * activation(W1*x + b1) + b2
        return self.linear2(self.activation(self.linear1(x)))

# Typical transformer: 4x expansion
d_model = 768
d_ff = d_model * 4  # 3072

ffn = FFN(d_model, d_ff)
x = torch.randn(2, 10, d_model)  # batch=2, seq=10

output = ffn(x)
print(f"Input shape: {x.shape}")
print(f"Hidden (expanded): {d_ff}")
print(f"Output shape: {output.shape}")

# Count parameters
n_params = sum(p.numel() for p in ffn.parameters())
print(f"\nFFN Parameters: {n_params:,}")
print(f"  Linear1: {d_model} x {d_ff} + {d_ff} = {d_model * d_ff + d_ff:,}")
print(f"  Linear2: {d_ff} x {d_model} + {d_model} = {d_ff * d_model + d_model:,}")
print("\nFFN contains most of the transformer's parameters!")
```

**Output:**

```
Input shape: torch.Size([2, 10, 768])
Hidden (expanded): 3072
Output shape: torch.Size([2, 10, 768])

FFN Parameters: 4,722,432
  Linear1: 768 x 3072 + 3072 = 2,362,368
  Linear2: 3072 x 768 + 768 = 2,360,064

FFN contains most of the transformer's parameters!
```

---

## 📏 Layer Normalization Flow

- Purpose: stabilize training, faster convergence
- LayerNorm: x_hat = (x - mu)/sqrt(var + eps)
- mu = mean(x), var = var(x) across features
- Output: y = gamma * x_hat + beta (learnable scale/shift)
- Pre-norm: LN(x) -> Attention -> residual
- Post-norm: x -> Attention -> LN -> residual
- Modern: Pre-norm is more stable
- Gradient: dL/dx involves mean and variance
- Normalization -> prevents gradient explosion
- Each layer gets normalized inputs
- Critical for deep transformer training

### Code Examples

#### Layer Norm

*Layer normalization normalizes across features, stabilizing training and enabling deeper networks.*

```python
import torch
import torch.nn as nn

# Layer Normalization: normalize across features
batch_size, seq_len, d_model = 2, 4, 8

x = torch.randn(batch_size, seq_len, d_model) * 10 + 5  # Non-normalized input

print("=== Before LayerNorm ===")
print(f"Mean: {x.mean().item():.4f}")
print(f"Std:  {x.std().item():.4f}")
print(f"Sample values: {x[0, 0, :4].tolist()}")

# Apply LayerNorm
ln = nn.LayerNorm(d_model)
y = ln(x)

print("\n=== After LayerNorm ===")
print(f"Mean: {y.mean().item():.4f} (close to 0)")
print(f"Std:  {y.std().item():.4f} (close to 1)")
print(f"Sample values: {[f'{v:.3f}' for v in y[0, 0, :4].tolist()]}")

# Manual computation
print("\n=== Manual LayerNorm ===")
x_sample = x[0, 0]  # One position
mean = x_sample.mean()
var = x_sample.var(unbiased=False)
eps = 1e-5
x_norm = (x_sample - mean) / torch.sqrt(var + eps)
print(f"Manual mean: {x_norm.mean().item():.6f}")
print(f"Manual std:  {x_norm.std(unbiased=False).item():.6f}")
```

**Output:**

```
=== Before LayerNorm ===
Mean: 3.1963
Std:  8.2841
Sample values: [8.709091186523438, 2.7072408199310303, 4.932647228240967, 5.221533298492432]

=== After LayerNorm ===
Mean: 0.0000 (close to 0)
Std:  1.0079 (close to 1)
Sample values: ['0.806', '-0.198', '0.174', '0.223']

=== Manual LayerNorm ===
Manual mean: -0.000000
Manual std:  1.000000
```

---

## 📍 Positional Encoding

- Problem: Attention has no position awareness
- Solution: Add positional information to inputs
- Sinusoidal: PE(pos,2i) = sin(pos/10000^(2i/d))
- PE(pos,2i+1) = cos(pos/10000^(2i/d))
- Properties: unique for each position
- Relative positions: sin(a+b) = sin(a)cos(b)+...
- Learned PE: trainable position embeddings
- Rotary PE: multiply complex rotations
- Position encoding is added, not concatenated
- Enables length generalization
- Different frequencies capture different scales

### Code Examples

#### Positional Encoding

*Positional encoding adds position information using sinusoids at different frequencies.*

```python
import torch
import math

def positional_encoding(max_len, d_model):
    '''Sinusoidal positional encoding'''
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()

    # div_term = 10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

    return pe

# Generate positional encodings
max_len, d_model = 10, 8
pe = positional_encoding(max_len, d_model)

print(f"Positional Encoding shape: {pe.shape}")
print(f"\nPosition 0: {[f'{v:.3f}' for v in pe[0].tolist()]}")
print(f"Position 1: {[f'{v:.3f}' for v in pe[1].tolist()]}")
print(f"Position 9: {[f'{v:.3f}' for v in pe[9].tolist()]}")

# Each position has unique encoding
print(f"\nAll positions unique: {len(set(tuple(row.tolist()) for row in pe)) == max_len}")

# Low frequencies vary slowly, high frequencies vary quickly
print(f"\nDim 0 (low freq) across positions: {[f'{pe[i, 0].item():.2f}' for i in range(5)]}")
print(f"Dim 6 (high freq) across positions: {[f'{pe[i, 6].item():.2f}' for i in range(5)]}")
```

**Output:**

```
Positional Encoding shape: torch.Size([10, 8])

Position 0: ['0.000', '1.000', '0.000', '1.000', '0.000', '1.000', '0.000', '1.000']
Position 1: ['0.841', '0.540', '0.100', '0.995', '0.010', '1.000', '0.001', '1.000']
Position 9: ['0.412', '-0.911', '0.783', '0.622', '0.090', '0.996', '0.009', '1.000']

All positions unique: True

Dim 0 (low freq) across positions: ['0.00', '0.84', '0.91', '0.14', '-0.76']
Dim 6 (high freq) across positions: ['0.00', '0.00', '0.00', '0.00', '0.00']
```

---

## ↩️ Gradient Flow & Backprop

- Transformer gradient flow is complex but clean:
- Loss -> Softmax -> Attention -> Residual -> Input
- Chain rule: dL/dx = dL/dy * dy/dx (compose)
- Residual connections: dL/dx = dL/dy * (1 + dF/dx)
- Multi-head: gradients split across heads
- Attention gradient: through softmax (complex)
- d_alpha_ij/d_e_ik = alpha_ij(delta_jk - alpha_ik)
- Layer norm: involves mean/variance gradients
- FFN: standard backprop through layers
- Residuals prevent vanishing gradients
- Layer norm prevents exploding gradients

### Code Examples

#### Gradient Flow

*Residual connections ensure gradients can flow directly backward, preventing vanishing gradients.*

```python
import torch
import torch.nn as nn

# Demonstrate gradient flow with residual connections
class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Residual: y = x + F(x)
        # Gradient: dy/dx = 1 + dF/dx (always at least 1!)
        return x + self.linear(x)

class NoResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x)

d_model = 64
num_layers = 10

# With residual connections
residual_model = nn.Sequential(*[ResidualBlock(d_model) for _ in range(num_layers)])

# Without residual connections
no_residual_model = nn.Sequential(*[NoResidualBlock(d_model) for _ in range(num_layers)])

# Test gradient flow
x = torch.randn(1, d_model, requires_grad=True)
y_res = residual_model(x).sum()
y_res.backward()
grad_res = x.grad.norm().item()

x = torch.randn(1, d_model, requires_grad=True)
y_no_res = no_residual_model(x).sum()
y_no_res.backward()
grad_no_res = x.grad.norm().item()

print(f"Gradient norm through {num_layers} layers:")
print(f"  With residuals:    {grad_res:.4f}")
print(f"  Without residuals: {grad_no_res:.4f}")
print(f"\nResidual connections preserve gradient flow!")
```

**Output:**

```
Gradient norm through 10 layers:
  With residuals:    26.9715
  Without residuals: 0.0355

Residual connections preserve gradient flow!
```

---

## 🎓 Training Dynamics

- Loss landscape: high-dimensional, non-convex
- Adam optimizer: adaptive learning rates per param
- m_t = beta1*m_{t-1} + (1-beta1)*grad_L (momentum)
- v_t = beta2*v_{t-1} + (1-beta2)*grad_L^2 (variance)
- theta_{t+1} = theta_t - alpha*m_hat_t/sqrt(v_hat_t + eps)
- Learning rate schedule: warmup then decay
- Warmup: prevent large updates early
- Weight decay: L_total = L + lambda*||theta||^2
- Gradient clipping: prevent explosion
- Batch size: affects gradient noise
- Critical: proper initialization (Xavier/He)

### Code Examples

#### Adam Optimizer

*Adam combines momentum and adaptive learning rates for efficient optimization.*

```python
import torch
import torch.nn as nn

# Adam optimizer: adaptive learning rates
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

print("Adam Optimizer Components:")
print(f"  Learning rate (alpha): {optimizer.defaults['lr']}")
print(f"  Beta1 (momentum): {optimizer.defaults['betas'][0]}")
print(f"  Beta2 (variance): {optimizer.defaults['betas'][1]}")
print(f"  Epsilon: {optimizer.defaults['eps']}")

# Simulate training steps
x = torch.randn(32, 10)
target = torch.randn(32, 1)

print("\nTraining steps:")
for step in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()

    # Before step: check gradient
    grad_norm = model.weight.grad.norm().item()

    optimizer.step()

    print(f"  Step {step+1}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

print("\nAdam adapts learning rate per parameter based on gradient history!")
```

**Output:**

```
Adam Optimizer Components:
  Learning rate (alpha): 0.001
  Beta1 (momentum): 0.9
  Beta2 (variance): 0.999
  Epsilon: 1e-08

Training steps:
  Step 1: loss=1.6536, grad_norm=2.3204
  Step 2: loss=1.6471, grad_norm=2.3087
  Step 3: loss=1.6407, grad_norm=2.2970
  Step 4: loss=1.6343, grad_norm=2.2854
  Step 5: loss=1.6279, grad_norm=2.2737

Adam adapts learning rate per parameter based on gradient history!
```

---

## 👁️ Attention Patterns

- What attention learns to do:
- Local patterns: syntax, grammar rules
- Long-range: coreference, semantic relations
- Attention heads specialize: syntax vs semantics
- Early layers: local patterns (POS, syntax)
- Later layers: global semantics, reasoning
- Attention weights: interpretable (somewhat)
- Causal masking: mask_ij = -inf if i < j
- Self-attention: each token attends to all
- Cross-attention: attend to different sequence
- Attention is all you need: no recurrence!

### Code Examples

#### Causal Mask

*Causal masking prevents the model from attending to future tokens during generation.*

```python
import torch
import torch.nn.functional as F

# Causal (autoregressive) masking for decoder
seq_len = 5

# Create causal mask: can only attend to past positions
# mask[i,j] = True means position i CAN attend to position j
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
print("Causal Mask (1 = can attend):")
print(causal_mask.int())

# In practice, we use -inf for positions we can't attend to
attn_mask = torch.zeros(seq_len, seq_len)
attn_mask = attn_mask.masked_fill(~causal_mask, float('-inf'))
print("\nAttention Mask (for softmax):")
print(attn_mask)

# Example: attention scores before masking
scores = torch.randn(seq_len, seq_len)
print("\nRaw Scores:")
print(scores.round(decimals=2))

# Apply mask and softmax
masked_scores = scores + attn_mask
attn_weights = F.softmax(masked_scores, dim=-1)
print("\nAttention Weights (after mask + softmax):")
print(attn_weights.round(decimals=2))
print("\nNotice: each row only attends to current and past positions!")
```

**Output:**

```
Causal Mask (1 = can attend):
tensor([[1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=torch.int32)

Attention Mask (for softmax):
tensor([[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]])

Raw Scores:
tensor([[ 0.6800,  0.7200, -1.0600, -2.1100,  0.1100],
        [-1.4200,  0.1100, -0.2300,  0.3600, -1.4400],
        [-0.8300, -2.8300, -0.5900,  1.8300,  0.7600],
        [-0.0800,  0.7000,  0.4000, -1.1400,  1.7800],
        [-0.7100, -0.4400,  0.1600, -1.5400, -0.3900]])

Attention Weights (after mask + softmax):
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1800, 0.8200, 0.0000, 0.0000, 0.0000],
        [0.4200, 0.0600, 0.5300, 0.0000, 0.0000],
        [0.1900, 0.4300, 0.3100, 0.0700, 0.0000],
        [0.1500, 0.2000, 0.3700, 0.0700, 0.2100]])

Notice: each row only attends to current and past positions!
```

---

## 💎 Embeddings & Representations

- Token embedding: discrete -> continuous
- Embedding matrix: V x d (vocab x dimension)
- Lookup: x_embed = E[token_id]
- Position-aware: x = x_embed + x_pos
- Contextual: same word, different meanings
- Deep representations: hierarchical features
- Early layers: syntax, low-level features
- Deep layers: semantics, high-level concepts
- Representation learning: unsupervised
- Transfer learning: pre-train -> fine-tune
- Embedding space: semantic similarity

### Code Examples

#### Embedding Lookup

*Embeddings convert discrete token IDs into continuous vectors that the model can process.*

```python
import torch
import torch.nn as nn

# Token Embedding: discrete tokens -> continuous vectors
vocab_size = 1000
d_model = 64

embedding = nn.Embedding(vocab_size, d_model)

# Token IDs (e.g., from tokenizer)
token_ids = torch.tensor([[42, 123, 7, 500],
                          [99, 42, 200, 1]])  # batch=2, seq=4

print(f"Token IDs shape: {token_ids.shape}")
print(f"Token IDs:\n{token_ids}")

# Lookup embeddings
embedded = embedding(token_ids)
print(f"\nEmbedded shape: {embedded.shape}")

# Same token = same embedding
print(f"\nToken 42 appears twice. Same embedding?")
print(f"  Position (0,0): {embedded[0, 0, :4].tolist()}")
print(f"  Position (1,1): {embedded[1, 1, :4].tolist()}")
print(f"  Equal: {torch.equal(embedded[0, 0], embedded[1, 1])}")

# Embedding matrix
print(f"\nEmbedding matrix shape: {embedding.weight.shape}")
print(f"Total parameters: {vocab_size * d_model:,}")
```

**Output:**

```
Token IDs shape: torch.Size([2, 4])
Token IDs:
tensor([[ 42, 123,   7, 500],
        [ 99,  42, 200,   1]])

Embedded shape: torch.Size([2, 4, 64])

Token 42 appears twice. Same embedding?
  Position (0,0): [-0.4602436423301697, -0.7743330597877502, -2.0495150089263916, 0.21481771767139435]
  Position (1,1): [-0.4602436423301697, -0.7743330597877502, -2.0495150089263916, 0.21481771767139435]
  Equal: True

Embedding matrix shape: torch.Size([1000, 64])
Total parameters: 64,000
```

---

## 🎭 Masked Language Modeling

- Pre-training objective: predict masked tokens
- Input: 'The cat [MASK] on the mat'
- Target: predict 'sat' for [MASK]
- Bidirectional: use both left and right context
- Loss: CrossEntropy over vocabulary
- Gradient: flows back through entire model
- Self-supervised: no human labels needed
- Massive datasets: learn language patterns
- Contextual understanding: 'bank' (river/money)
- Next sentence prediction: sentence relationships
- MLM -> powerful representations

---

## ✍️ Autoregressive Generation

- Causal/Decoder model: predict next token
- Input: 'The cat sat' -> predict 'on'
- Causal mask: can't see future tokens
- Generation: sample from output distribution
- Temperature: p'_i = p_i^(1/T) (control randomness)
- Beam search: keep top-k sequences
- Top-k sampling: sample from top k tokens
- Nucleus sampling: cumulative probability p
- Length penalty: prevent short sequences
- Teacher forcing: use ground truth during training
- Inference: autoregressive token by token

### Code Examples

#### Temperature Sampling

*Temperature scaling controls the randomness of sampling: lower is more deterministic, higher is more random.*

```python
import torch
import torch.nn.functional as F

# Temperature controls randomness in generation
logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -0.5])
print(f"Logits: {logits.tolist()}")

temperatures = [0.5, 1.0, 2.0]

print("\nProbabilities at different temperatures:")
for temp in temperatures:
    # Apply temperature: logits / T
    scaled_logits = logits / temp
    probs = F.softmax(scaled_logits, dim=-1)
    print(f"  T={temp}: {[f'{p:.3f}' for p in probs.tolist()]}")

print("\nLower T -> more confident (peaky)")
print("Higher T -> more uniform (diverse)")

# Sampling demonstration
print("\nSampling 10 tokens at each temperature:")
for temp in temperatures:
    scaled_logits = logits / temp
    probs = F.softmax(scaled_logits, dim=-1)
    samples = torch.multinomial(probs, num_samples=10, replacement=True)
    print(f"  T={temp}: {samples.tolist()}")
```

**Output:**

```
Logits: [2.0, 1.0, 0.5, 0.10000000149011612, -0.5]

Probabilities at different temperatures:
  T=0.5: ['0.824', '0.111', '0.041', '0.018', '0.006']
  T=1.0: ['0.549', '0.202', '0.122', '0.082', '0.045']
  T=2.0: ['0.363', '0.220', '0.172', '0.141', '0.104']

Lower T -> more confident (peaky)
Higher T -> more uniform (diverse)

Sampling 10 tokens at each temperature:
  T=0.5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  T=1.0: [0, 2, 3, 0, 0, 1, 1, 0, 0, 1]
  T=2.0: [2, 0, 1, 1, 2, 1, 2, 0, 3, 0]
```

---

## 📈 Scale & Emergence

- Scaling laws: performance ~ compute^alpha
- Parameters: 100M -> 1B -> 100B -> 1T+
- Emergent abilities: appear at scale
- In-context learning: few-shot without training
- Chain of thought: step-by-step reasoning
- Grokking: sudden generalization
- Compute optimal: Chinchilla scaling
- Data scaling: more data = better models
- Model parallel: split across GPUs
- Gradient checkpointing: memory vs compute
- Scale changes everything!

---

## 🔧 Optimization Tricks

- Gradient accumulation: simulate large batches
- Mixed precision: FP16 forward, FP32 backward
- Dynamic loss scaling: prevent underflow
- Gradient clipping: ||g|| > threshold
- Learning rate schedules: cosine, polynomial
- Weight decay: different for different params
- Layer-wise learning rates: different per layer
- Dropout: prevent overfitting
- Label smoothing: soften target distribution
- Data augmentation: increase diversity
- All tricks matter at scale!

---

## 🚀 Modern Improvements

- RMSNorm: simpler than LayerNorm
- SwiGLU: better activation for transformers
- Rotary Position Embedding: better positions
- Flash Attention: memory-efficient attention
- Gradient checkpointing: save memory
- Mixture of Experts: conditional computation
- Sparse attention: reduce O(n^2) complexity
- Linear attention: approximate attention
- Group Query Attention: fewer KV heads
- Multi-Query Attention: shared K,V
- Continuous innovation in architectures!

---
