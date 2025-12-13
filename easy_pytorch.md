# Easy PyTorch v1.0

**An Essential Guide for Beginners**

*by jw*

This book covers the fundamental PyTorch functions organized into categories. Each function includes:

- **Code example** - Working Python code
- **Output** - Actual execution result
- **Explanation** - Clear, concise description

*Based on PyTorch 2.8.0*

---

## Table of Contents

0. [Hello World](#hello-world)
1. [Tensor Creation (torch.*)](#tensor-creation-(torch.*))
2. [Basic Operations (torch.*)](#basic-operations-(torch.*))
3. [Shape Manipulation (x.*)](#shape-manipulation-(x.*))
4. [Indexing & Slicing (mixed)](#indexing--slicing-(mixed))
5. [Reduction Ops (x.*)](#reduction-ops-(x.*))
6. [Math Functions (torch.*)](#math-functions-(torch.*))
7. [Linear Algebra (torch.*)](#linear-algebra-(torch.*))
8. [Neural Network (F.*)](#neural-network-(f.*))
9. [Loss Functions (F.*)](#loss-functions-(f.*))
10. [Pooling & Conv (F.*)](#pooling--conv-(f.*))
11. [Advanced Ops (torch.*)](#advanced-ops-(torch.*))
12. [Autograd (mixed)](#autograd-(mixed))
13. [Device Ops (mixed)](#device-ops-(mixed))
14. [Utilities (mixed)](#utilities-(mixed))
15. [Comparison (torch.*)](#comparison-(torch.*))
16. [Tensor Methods (x.*)](#tensor-methods-(x.*))

---

## Hello World

This chapter shows a **complete** neural network in one simple example. Let's teach a tiny neural network to learn the XOR function.

### What is XOR?

| Input A | Input B | Output (A XOR B) |
|---------|---------|------------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### 4 Steps

1. **Data** - Define input/output pairs
2. **Model** - Create neural network
3. **Training** - Learn from data (forward → loss → backward → update)
4. **Inference** - Make predictions

### Complete Code

```python
import torch
import torch.nn as nn

# === HELLO WORLD: Tiny Neural Network ===
# Goal: Learn XOR function (0^0=0, 0^1=1, 1^0=1, 1^1=0)

# 1. DATA - tiny input/output pairs
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

print("=== Data ===")
print(f"Input X:\n{X}")
print(f"Target y: {y.flatten().tolist()}")

# 2. MODEL - simple 2-layer network
model = nn.Sequential(
    nn.Linear(2, 4),   # 2 inputs -> 4 hidden
    nn.ReLU(),         # activation
    nn.Linear(4, 1),   # 4 hidden -> 1 output
    nn.Sigmoid()       # output 0~1
)

print(f"\n=== Model ===")
print(model)

# 3. TRAINING
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

print(f"\n=== Training ===")
for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}, Loss: {loss.item():.4f}")

# 4. INFERENCE
print(f"\n=== Inference ===")
with torch.no_grad():
    predictions = model(X)
    rounded = (predictions > 0.5).float()

print("Input -> Predicted -> Rounded -> Target")
for i in range(4):
    inp = X[i].tolist()
    pred_val = predictions[i].item()
    round_val = int(rounded[i].item())
    target = int(y[i].item())
    status = "OK" if round_val == target else "WRONG"
    print(f"{inp} -> {pred_val:.3f} -> {round_val} -> {target} {status}")

accuracy = (rounded == y).float().mean()
print(f"\nAccuracy: {accuracy.item()*100:.0f}%")
```

### Output

```
=== Data ===
Input X:
tensor([[0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]])
Target y: [0.0, 1.0, 1.0, 0.0]

=== Model ===
Sequential(
  (0): Linear(in_features=2, out_features=4, bias=True)
  (1): ReLU()
  (2): Linear(in_features=4, out_features=1, bias=True)
  (3): Sigmoid()
)

=== Training ===
Epoch    0, Loss: 0.2455
Epoch  200, Loss: 0.1671
Epoch  400, Loss: 0.1668
Epoch  600, Loss: 0.1668
Epoch  800, Loss: 0.1667

=== Inference ===
Input -> Predicted -> Rounded -> Target
[0.0, 0.0] -> 0.334 -> 0 -> 0 OK
[0.0, 1.0] -> 0.988 -> 1 -> 1 OK
[1.0, 0.0] -> 0.334 -> 0 -> 1 WRONG
[1.0, 1.0] -> 0.334 -> 0 -> 0 OK

Accuracy: 75%
```

### Key Concepts

- `nn.Sequential` - Stack layers sequentially
- `nn.Linear(in, out)` - Fully connected layer
- `nn.ReLU()` - Activation function
- `nn.MSELoss()` - Loss function (how wrong?)
- `optimizer.zero_grad()` - Reset gradients
- `loss.backward()` - Compute gradients (backprop)
- `optimizer.step()` - Update weights
- `torch.no_grad()` - Disable gradients for inference

**Done!** You've trained your first neural network.

---

## ■ Tensor Creation (torch.*)

#### `torch.tensor(data)`

*Creates a tensor from data (list, numpy array, etc.). Infers dtype automatically.*

**Example:**

```python
import torch
# Create tensor from Python list
x = torch.tensor([[1, 2], [3, 4]])
print("Tensor from list:")
print(x)
print(f"Shape: {x.shape}, dtype: {x.dtype}")
```

**Output:**

```
Tensor from list:
tensor([[1, 2],
        [3, 4]])
Shape: torch.Size([2, 2]), dtype: torch.int64
```

#### `torch.zeros(*size)`

*Creates a tensor filled with zeros. Useful for initialization.*

**Example:**

```python
import torch
# Create tensor filled with zeros
x = torch.zeros(2, 3)
print("Zeros tensor (2x3):")
print(x)
```

**Output:**

```
Zeros tensor (2x3):
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

#### `torch.ones(*size)`

*Creates a tensor filled with ones. Often used for masks or initialization.*

**Example:**

```python
import torch
# Create tensor filled with ones
x = torch.ones(3, 2)
print("Ones tensor (3x2):")
print(x)
```

**Output:**

```
Ones tensor (3x2):
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
```

#### `torch.eye(n)  # identity matrix`

*Creates an identity matrix (1s on diagonal, 0s elsewhere). Used in linear algebra.*

**Example:**

```python
import torch
# Create identity matrix
x = torch.eye(3)
print("Identity matrix (3x3):")
print(x)
```

**Output:**

```
Identity matrix (3x3):
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
```

#### `torch.arange(start, end, step)`

*Creates 1D tensor with evenly spaced values. Similar to Python's range().*

**Example:**

```python
import torch
# Create range tensor
x = torch.arange(0, 10, 2)
print("Range [0, 10) step 2:")
print(x)
```

**Output:**

```
Range [0, 10) step 2:
tensor([0, 2, 4, 6, 8])
```

#### `torch.linspace(start, end, steps)`

*Creates tensor with specified number of equally spaced points between start and end.*

**Example:**

```python
import torch
# Create linearly spaced tensor
x = torch.linspace(0, 1, 5)
print("Linspace 0 to 1, 5 points:")
print(x)
```

**Output:**

```
Linspace 0 to 1, 5 points:
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

#### `torch.logspace(start, end, steps)`

*Creates tensor with logarithmically spaced values. Useful for learning rates.*

**Example:**

```python
import torch
# Create logarithmically spaced tensor
x = torch.logspace(0, 2, 3)  # 10^0, 10^1, 10^2
print("Logspace 10^0 to 10^2:")
print(x)
```

**Output:**

```
Logspace 10^0 to 10^2:
tensor([  1.,  10., 100.])
```

#### `torch.rand(*size)  # uniform [0,1)`

*Creates tensor with uniform random values between 0 and 1.*

**Example:**

```python
import torch
torch.manual_seed(42)
# Create random tensor [0, 1)
x = torch.rand(2, 3)
print("Random uniform [0,1):")
print(x)
```

**Output:**

```
Random uniform [0,1):
tensor([[0.8823, 0.9150, 0.3829],
        [0.9593, 0.3904, 0.6009]])
```

#### `torch.randn(*size)  # normal N(0,1)`

*Creates tensor with values from standard normal distribution (mean=0, std=1).*

**Example:**

```python
import torch
torch.manual_seed(42)
# Create random tensor from normal distribution
x = torch.randn(2, 3)
print("Random normal N(0,1):")
print(x)
```

**Output:**

```
Random normal N(0,1):
tensor([[ 0.3367,  0.1288,  0.2345],
        [ 0.2303, -1.1229, -0.1863]])
```

#### `torch.randint(low, high, size)`

*Creates tensor with random integers in [low, high) range.*

**Example:**

```python
import torch
torch.manual_seed(42)
# Create random integers
x = torch.randint(0, 10, (2, 3))
print("Random integers [0, 10):")
print(x)
```

**Output:**

```
Random integers [0, 10):
tensor([[2, 7, 6],
        [4, 6, 5]])
```

#### `torch.empty(*size)`

*Creates uninitialized tensor. Faster than zeros/ones but contains garbage values.*

**Example:**

```python
import torch
# Create uninitialized tensor
x = torch.empty(2, 2)
print("Empty tensor (uninitialized):")
print(x)
print("Warning: Contains garbage values!")
```

**Output:**

```
Empty tensor (uninitialized):
tensor([[0., 0.],
        [0., 0.]])
Warning: Contains garbage values!
```

#### `torch.full(size, fill_value)`

*Creates tensor filled with a specific value. Useful for constants.*

**Example:**

```python
import torch
# Create tensor filled with specific value
x = torch.full((2, 3), 7.0)
print("Tensor filled with 7.0:")
print(x)
```

**Output:**

```
Tensor filled with 7.0:
tensor([[7., 7., 7.],
        [7., 7., 7.]])
```

#### `torch.zeros_like(x), ones_like(x)`

*Creates zeros/ones tensor matching shape and dtype of input tensor.*

**Example:**

```python
import torch
original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
zeros = torch.zeros_like(original)
ones = torch.ones_like(original)
print("Original:", original.shape)
print("Zeros like:", zeros)
print("Ones like:", ones)
```

**Output:**

```
Original: torch.Size([2, 2])
Zeros like: tensor([[0., 0.],
        [0., 0.]])
Ones like: tensor([[1., 1.],
        [1., 1.]])
```

---

## ⚙ Basic Operations (torch.*)

#### `torch.add(a, b) or a + b`

*Element-wise addition. Supports broadcasting for different shapes.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
result = torch.add(a, b)
print(f"{a} + {b} = {result}")
```

**Output:**

```
tensor([1, 2, 3]) + tensor([4, 5, 6]) = tensor([5, 7, 9])
```

#### `torch.sub(a, b) or a - b`

*Element-wise subtraction. Supports broadcasting.*

**Example:**

```python
import torch
a = torch.tensor([5, 6, 7])
b = torch.tensor([1, 2, 3])
result = torch.sub(a, b)
print(f"{a} - {b} = {result}")
```

**Output:**

```
tensor([5, 6, 7]) - tensor([1, 2, 3]) = tensor([4, 4, 4])
```

#### `torch.mul(a, b) or a * b`

*Element-wise multiplication (Hadamard product). Not matrix multiplication.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
result = torch.mul(a, b)
print(f"{a} * {b} = {result}")
```

**Output:**

```
tensor([1, 2, 3]) * tensor([4, 5, 6]) = tensor([ 4, 10, 18])
```

#### `torch.div(a, b) or a / b`

*Element-wise division. Use float tensors to avoid integer division.*

**Example:**

```python
import torch
a = torch.tensor([10.0, 20.0, 30.0])
b = torch.tensor([2.0, 4.0, 5.0])
result = torch.div(a, b)
print(f"{a} / {b} = {result}")
```

**Output:**

```
tensor([10., 20., 30.]) / tensor([2., 4., 5.]) = tensor([5., 5., 6.])
```

#### `torch.matmul(a, b) or a @ b`

*Matrix multiplication. For 2D tensors, performs standard matrix multiply.*

**Example:**

```python
import torch
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(a, b)
print("Matrix A:")
print(a)
print("Matrix B:")
print(b)
print("A @ B:")
print(result)
```

**Output:**

```
Matrix A:
tensor([[1, 2],
        [3, 4]])
Matrix B:
tensor([[5, 6],
        [7, 8]])
A @ B:
tensor([[19, 22],
        [43, 50]])
```

#### `torch.pow(a, exp) or a ** exp`

*Element-wise power operation. Can use scalar or tensor exponent.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
result = torch.pow(x, 2)
print(f"{x} ** 2 = {result}")
```

**Output:**

```
tensor([1., 2., 3.]) ** 2 = tensor([1., 4., 9.])
```

#### `torch.abs(x)  # absolute value`

*Returns absolute value of each element.*

**Example:**

```python
import torch
x = torch.tensor([-1, -2, 3, -4])
result = torch.abs(x)
print(f"abs({x}) = {result}")
```

**Output:**

```
abs(tensor([-1, -2,  3, -4])) = tensor([1, 2, 3, 4])
```

#### `torch.neg(x)  # negative`

*Returns negation of each element. Same as -x.*

**Example:**

```python
import torch
x = torch.tensor([1, -2, 3])
result = torch.neg(x)
print(f"neg({x}) = {result}")
```

**Output:**

```
neg(tensor([ 1, -2,  3])) = tensor([-1,  2, -3])
```

#### `torch.reciprocal(x)  # 1/x`

*Returns reciprocal (1/x) of each element.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 4.0])
result = torch.reciprocal(x)
print(f"1/{x} = {result}")
```

**Output:**

```
1/tensor([1., 2., 4.]) = tensor([1.0000, 0.5000, 0.2500])
```

#### `torch.remainder(a, b)  # remainder`

*Element-wise remainder (modulo operation).*

**Example:**

```python
import torch
a = torch.tensor([10, 11, 12])
b = torch.tensor([3, 3, 3])
result = torch.remainder(a, b)
print(f"{a} % {b} = {result}")
```

**Output:**

```
tensor([10, 11, 12]) % tensor([3, 3, 3]) = tensor([1, 2, 0])
```

---

## ↻ Shape Manipulation (x.*)

#### `x.reshape(*shape)`

*Returns tensor with new shape. Total elements must match. May copy data.*

**Example:**

```python
import torch
x = torch.arange(6)
print(f"Original: {x}")
reshaped = x.reshape(2, 3)
print("Reshaped to (2, 3):")
print(reshaped)
```

**Output:**

```
Original: tensor([0, 1, 2, 3, 4, 5])
Reshaped to (2, 3):
tensor([[0, 1, 2],
        [3, 4, 5]])
```

#### `x.view(*shape)`

*Returns view with new shape. Requires contiguous memory. Shares data.*

**Example:**

```python
import torch
x = torch.arange(12)
print(f"Original: {x}")
viewed = x.view(3, 4)
print("View as (3, 4):")
print(viewed)
```

**Output:**

```
Original: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
View as (3, 4):
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```

#### `x.transpose(dim0, dim1)`

*Swaps two dimensions. For 2D, equivalent to matrix transpose.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Original (2x3):")
print(x)
transposed = x.transpose(0, 1)
print("Transposed (3x2):")
print(transposed)
```

**Output:**

```
Original (2x3):
tensor([[1, 2, 3],
        [4, 5, 6]])
Transposed (3x2):
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

#### `x.permute(*dims)`

*Reorders all dimensions. More flexible than transpose.*

**Example:**

```python
import torch
x = torch.randn(2, 3, 4)
print(f"Original shape: {x.shape}")
permuted = x.permute(2, 0, 1)
print(f"Permuted shape: {permuted.shape}")
```

**Output:**

```
Original shape: torch.Size([2, 3, 4])
Permuted shape: torch.Size([4, 2, 3])
```

#### `x.squeeze(dim)`

*Removes dimensions of size 1. Reduces tensor rank.*

**Example:**

```python
import torch
x = torch.zeros(1, 3, 1, 4)
print(f"Original shape: {x.shape}")
squeezed = x.squeeze()
print(f"Squeezed shape: {squeezed.shape}")
```

**Output:**

```
Original shape: torch.Size([1, 3, 1, 4])
Squeezed shape: torch.Size([3, 4])
```

#### `x.unsqueeze(dim)`

*Adds dimension of size 1 at specified position.*

**Example:**

```python
import torch
x = torch.tensor([1, 2, 3])
print(f"Original shape: {x.shape}")
unsqueezed = x.unsqueeze(0)
print(f"Unsqueezed at dim 0: {unsqueezed.shape}")
print(unsqueezed)
```

**Output:**

```
Original shape: torch.Size([3])
Unsqueezed at dim 0: torch.Size([1, 3])
tensor([[1, 2, 3]])
```

#### `x.flatten(start_dim, end_dim)`

*Flattens tensor into 1D. Optional start/end dims for partial flatten.*

**Example:**

```python
import torch
x = torch.randn(2, 3, 4)
print(f"Original shape: {x.shape}")
flat = x.flatten()
print(f"Flattened: {flat.shape}")
```

**Output:**

```
Original shape: torch.Size([2, 3, 4])
Flattened: torch.Size([24])
```

#### `x.expand(*sizes)`

*Expands tensor by repeating along size-1 dimensions. No data copy.*

**Example:**

```python
import torch
x = torch.tensor([[1], [2], [3]])
print(f"Original: {x.shape}")
expanded = x.expand(3, 4)
print("Expanded to (3, 4):")
print(expanded)
```

**Output:**

```
Original: torch.Size([3, 1])
Expanded to (3, 4):
tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]])
```

#### `x.repeat(*sizes)`

*Repeats tensor along each dimension. Creates new memory.*

**Example:**

```python
import torch
x = torch.tensor([1, 2])
print(f"Original: {x}")
repeated = x.repeat(3)
print(f"Repeated 3x: {repeated}")
```

**Output:**

```
Original: tensor([1, 2])
Repeated 3x: tensor([1, 2, 1, 2, 1, 2])
```

#### `x.contiguous()`

*Returns contiguous tensor in memory. Required before view() after transpose.*

**Example:**

```python
import torch
x = torch.randn(3, 4)
y = x.transpose(0, 1)
print(f"Is contiguous: {y.is_contiguous()}")
z = y.contiguous()
print(f"After contiguous(): {z.is_contiguous()}")
```

**Output:**

```
Is contiguous: False
After contiguous(): True
```

---

## ◉ Indexing & Slicing (mixed)

#### `x[i]  # index`

*Basic indexing returns row/element at index i.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor:")
print(x)
print(f"x[0] = {x[0]}")
print(f"x[1] = {x[1]}")
```

**Output:**

```
Tensor:
tensor([[1, 2, 3],
        [4, 5, 6]])
x[0] = tensor([1, 2, 3])
x[1] = tensor([4, 5, 6])
```

#### `x[i:j]  # slice`

*Slicing with start:stop:step. Works like Python lists.*

**Example:**

```python
import torch
x = torch.arange(10)
print(f"Original: {x}")
print(f"x[2:5] = {x[2:5]}")
print(f"x[::2] = {x[::2]}")
```

**Output:**

```
Original: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x[2:5] = tensor([2, 3, 4])
x[::2] = tensor([0, 2, 4, 6, 8])
```

#### `x[..., i]  # ellipsis`

*Ellipsis (...) represents all remaining dimensions.*

**Example:**

```python
import torch
x = torch.randn(2, 3, 4)
print(f"Shape: {x.shape}")
print(f"x[..., 0] shape: {x[..., 0].shape}")
print(f"x[0, ...] shape: {x[0, ...].shape}")
```

**Output:**

```
Shape: torch.Size([2, 3, 4])
x[..., 0] shape: torch.Size([2, 3])
x[0, ...] shape: torch.Size([3, 4])
```

#### `x[:, -1]  # last column`

*Negative indexing works from the end. -1 is last element.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor:")
print(x)
print(f"Last column x[:, -1] = {x[:, -1]}")
```

**Output:**

```
Tensor:
tensor([[1, 2, 3],
        [4, 5, 6]])
Last column x[:, -1] = tensor([3, 6])
```

#### `torch.index_select(x, dim, idx)`

*Selects elements along dimension using index tensor.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
idx = torch.tensor([0, 2])
result = torch.index_select(x, 0, idx)
print("Original:")
print(x)
print(f"Select rows {idx.tolist()}:")
print(result)
```

**Output:**

```
Original:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
Select rows [0, 2]:
tensor([[1, 2, 3],
        [7, 8, 9]])
```

#### `torch.masked_select(x, mask)`

*Returns 1D tensor of elements where mask is True.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
mask = x > 2
print(f"Tensor: {x}")
print(f"Mask (>2): {mask}")
print(f"Selected: {torch.masked_select(x, mask)}")
```

**Output:**

```
Tensor: tensor([[1, 2],
        [3, 4]])
Mask (>2): tensor([[False, False],
        [ True,  True]])
Selected: tensor([3, 4])
```

#### `torch.gather(x, dim, idx)  # gather`

*Gathers values along axis according to indices. Useful for selecting from distributions.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
idx = torch.tensor([[0, 0], [1, 0]])
result = torch.gather(x, 1, idx)
print("Original:")
print(x)
print(f"Gather with indices {idx.tolist()}:")
print(result)
```

**Output:**

```
Original:
tensor([[1, 2],
        [3, 4]])
Gather with indices [[0, 0], [1, 0]]:
tensor([[1, 1],
        [4, 3]])
```

#### `torch.scatter(x, dim, idx, src)`

*Writes values from src into x at positions specified by idx. Inverse of gather.*

**Example:**

```python
import torch
x = torch.zeros(3, 5)
idx = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
src = torch.ones(2, 5)
result = x.scatter(0, idx, src)
print("Scatter result:")
print(result)
```

**Output:**

```
Scatter result:
tensor([[1., 1., 1., 1., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.]])
```

#### `torch.where(cond, x, y)  # conditional`

*Returns elements from x where condition is True, else from y.*

**Example:**

```python
import torch
x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([10, 20, 30, 40, 50])
cond = x > 3
result = torch.where(cond, x, y)
print(f"x: {x}")
print(f"y: {y}")
print(f"where(x>3, x, y): {result}")
```

**Output:**

```
x: tensor([1, 2, 3, 4, 5])
y: tensor([10, 20, 30, 40, 50])
where(x>3, x, y): tensor([10, 20, 30,  4,  5])
```

#### `torch.take(x, indices)  # flat index`

*Treats tensor as 1D and returns elements at given indices.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
idx = torch.tensor([0, 2, 5])
result = torch.take(x, idx)
print(f"Tensor (flattened would be {x.flatten().tolist()})")
print(f"Take indices {idx.tolist()}: {result}")
```

**Output:**

```
Tensor (flattened would be [1, 2, 3, 4, 5, 6])
Take indices [0, 2, 5]: tensor([1, 3, 6])
```

---

## Σ Reduction Ops (x.*)

#### `x.sum(dim, keepdim)`

*Sums elements. Optional dim specifies axis. keepdim preserves dimensions.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor:")
print(x)
print(f"Sum all: {x.sum()}")
print(f"Sum dim=0: {x.sum(dim=0)}")
print(f"Sum dim=1: {x.sum(dim=1)}")
```

**Output:**

```
Tensor:
tensor([[1, 2, 3],
        [4, 5, 6]])
Sum all: 21
Sum dim=0: tensor([5, 7, 9])
Sum dim=1: tensor([ 6, 15])
```

#### `x.mean(dim, keepdim)`

*Computes mean. Requires float tensor.*

**Example:**

```python
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("Tensor:")
print(x)
print(f"Mean all: {x.mean()}")
print(f"Mean dim=1: {x.mean(dim=1)}")
```

**Output:**

```
Tensor:
tensor([[1., 2.],
        [3., 4.]])
Mean all: 2.5
Mean dim=1: tensor([1.5000, 3.5000])
```

#### `x.std(dim, unbiased)`

*Computes standard deviation. unbiased=True uses N-1 denominator.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Data: {x}")
print(f"Std (unbiased): {x.std():.4f}")
print(f"Std (biased): {x.std(unbiased=False):.4f}")
```

**Output:**

```
Data: tensor([1., 2., 3., 4., 5.])
Std (unbiased): 1.5811
Std (biased): 1.4142
```

#### `x.var(dim, unbiased)`

*Computes variance (std squared).*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Data: {x}")
print(f"Variance: {x.var():.4f}")
```

**Output:**

```
Data: tensor([1., 2., 3., 4., 5.])
Variance: 2.5000
```

#### `x.max(dim)  # values & indices`

*Returns max values and their indices along dimension.*

**Example:**

```python
import torch
x = torch.tensor([[1, 5, 3], [4, 2, 6]])
print("Tensor:")
print(x)
vals, idxs = x.max(dim=1)
print(f"Max per row: values={vals}, indices={idxs}")
```

**Output:**

```
Tensor:
tensor([[1, 5, 3],
        [4, 2, 6]])
Max per row: values=tensor([5, 6]), indices=tensor([1, 2])
```

#### `x.min(dim)  # values & indices`

*Returns min values and their indices along dimension.*

**Example:**

```python
import torch
x = torch.tensor([[1, 5, 3], [4, 2, 6]])
print("Tensor:")
print(x)
vals, idxs = x.min(dim=1)
print(f"Min per row: values={vals}, indices={idxs}")
```

**Output:**

```
Tensor:
tensor([[1, 5, 3],
        [4, 2, 6]])
Min per row: values=tensor([1, 2]), indices=tensor([0, 1])
```

#### `x.argmax(dim)  # indices only`

*Returns indices of maximum values. Commonly used with softmax output.*

**Example:**

```python
import torch
x = torch.tensor([[1, 5, 3], [4, 2, 6]])
print("Tensor:")
print(x)
print(f"Argmax (all): {x.argmax()}")
print(f"Argmax dim=1: {x.argmax(dim=1)}")
```

**Output:**

```
Tensor:
tensor([[1, 5, 3],
        [4, 2, 6]])
Argmax (all): 5
Argmax dim=1: tensor([1, 2])
```

#### `x.argmin(dim)  # indices only`

*Returns indices of minimum values.*

**Example:**

```python
import torch
x = torch.tensor([[1, 5, 3], [4, 2, 6]])
print("Tensor:")
print(x)
print(f"Argmin dim=1: {x.argmin(dim=1)}")
```

**Output:**

```
Tensor:
tensor([[1, 5, 3],
        [4, 2, 6]])
Argmin dim=1: tensor([0, 1])
```

#### `x.median(dim)`

*Returns median values and indices along dimension.*

**Example:**

```python
import torch
x = torch.tensor([[1, 5, 3], [4, 2, 6]])
print("Tensor:")
print(x)
vals, idxs = x.median(dim=1)
print(f"Median per row: values={vals}, indices={idxs}")
```

**Output:**

```
Tensor:
tensor([[1, 5, 3],
        [4, 2, 6]])
Median per row: values=tensor([3, 4]), indices=tensor([2, 0])
```

#### `x.mode(dim)`

*Returns most frequent values along dimension.*

**Example:**

```python
import torch
x = torch.tensor([[1, 1, 2], [3, 3, 3]])
print("Tensor:")
print(x)
vals, idxs = x.mode(dim=1)
print(f"Mode per row: values={vals}")
```

**Output:**

```
Tensor:
tensor([[1, 1, 2],
        [3, 3, 3]])
Mode per row: values=tensor([1, 3])
```

#### `x.prod(dim)  # product`

*Computes product of elements along dimension.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor:")
print(x)
print(f"Product all: {x.prod()}")
print(f"Product dim=1: {x.prod(dim=1)}")
```

**Output:**

```
Tensor:
tensor([[1, 2, 3],
        [4, 5, 6]])
Product all: 720
Product dim=1: tensor([  6, 120])
```

#### `x.cumsum(dim)  # cumulative sum`

*Cumulative sum along dimension. Each element is sum of all previous.*

**Example:**

```python
import torch
x = torch.tensor([1, 2, 3, 4])
print(f"Original: {x}")
print(f"Cumsum: {x.cumsum(dim=0)}")
```

**Output:**

```
Original: tensor([1, 2, 3, 4])
Cumsum: tensor([ 1,  3,  6, 10])
```

#### `x.norm(p, dim)  # Lp norm`

*Computes Lp norm. L2 (Euclidean) is default. L1 is Manhattan distance.*

**Example:**

```python
import torch
x = torch.tensor([3.0, 4.0])
print(f"Vector: {x}")
print(f"L2 norm: {x.norm():.4f}")  # sqrt(9+16) = 5
print(f"L1 norm: {x.norm(p=1):.4f}")  # 3+4 = 7
```

**Output:**

```
Vector: tensor([3., 4.])
L2 norm: 5.0000
L1 norm: 7.0000
```

---

## ∫ Math Functions (torch.*)

#### `torch.sin(x), cos(x), tan(x)`

*Trigonometric functions. Input in radians.*

**Example:**

```python
import torch
import math
x = torch.tensor([0, math.pi/2, math.pi])
print(f"x: {x}")
print(f"sin(x): {torch.sin(x)}")
print(f"cos(x): {torch.cos(x)}")
```

**Output:**

```
x: tensor([0.0000, 1.5708, 3.1416])
sin(x): tensor([ 0.0000e+00,  1.0000e+00, -8.7423e-08])
cos(x): tensor([ 1.0000e+00, -4.3711e-08, -1.0000e+00])
```

#### `torch.asin(x), acos(x), atan(x)`

*Inverse trigonometric functions. Returns radians.*

**Example:**

```python
import torch
x = torch.tensor([0.0, 0.5, 1.0])
print(f"x: {x}")
print(f"asin(x): {torch.asin(x)}")
print(f"acos(x): {torch.acos(x)}")
```

**Output:**

```
x: tensor([0.0000, 0.5000, 1.0000])
asin(x): tensor([0.0000, 0.5236, 1.5708])
acos(x): tensor([1.5708, 1.0472, 0.0000])
```

#### `torch.sinh(x), cosh(x), tanh(x)`

*Hyperbolic functions. tanh is commonly used as activation function.*

**Example:**

```python
import torch
x = torch.tensor([-1.0, 0.0, 1.0])
print(f"x: {x}")
print(f"tanh(x): {torch.tanh(x)}")
```

**Output:**

```
x: tensor([-1.,  0.,  1.])
tanh(x): tensor([-0.7616,  0.0000,  0.7616])
```

#### `torch.exp(x), log(x), log10(x)`

*Exponential and logarithmic functions. log is natural log (base e).*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
print(f"x: {x}")
print(f"exp(x): {torch.exp(x)}")
print(f"log(exp(x)): {torch.log(torch.exp(x))}")
```

**Output:**

```
x: tensor([1., 2., 3.])
exp(x): tensor([ 2.7183,  7.3891, 20.0855])
log(exp(x)): tensor([1.0000, 2.0000, 3.0000])
```

#### `torch.sqrt(x), rsqrt(x)`

*Square root and reciprocal square root (1/sqrt(x)).*

**Example:**

```python
import torch
x = torch.tensor([1.0, 4.0, 9.0])
print(f"x: {x}")
print(f"sqrt(x): {torch.sqrt(x)}")
print(f"rsqrt(x): {torch.rsqrt(x)}")  # 1/sqrt(x)
```

**Output:**

```
x: tensor([1., 4., 9.])
sqrt(x): tensor([1., 2., 3.])
rsqrt(x): tensor([1.0000, 0.5000, 0.3333])
```

#### `torch.floor(x), ceil(x), round(x)`

*Rounding functions. floor rounds down, ceil rounds up.*

**Example:**

```python
import torch
x = torch.tensor([1.2, 2.5, 3.7])
print(f"x: {x}")
print(f"floor(x): {torch.floor(x)}")
print(f"ceil(x): {torch.ceil(x)}")
print(f"round(x): {torch.round(x)}")
```

**Output:**

```
x: tensor([1.2000, 2.5000, 3.7000])
floor(x): tensor([1., 2., 3.])
ceil(x): tensor([2., 3., 4.])
round(x): tensor([1., 2., 4.])
```

#### `torch.clamp(x, min, max)`

*Clamps values to [min, max] range. Values outside are set to boundary.*

**Example:**

```python
import torch
x = torch.tensor([-2, 0, 3, 5, 10])
result = torch.clamp(x, min=0, max=5)
print(f"Original: {x}")
print(f"Clamped [0,5]: {result}")
```

**Output:**

```
Original: tensor([-2,  0,  3,  5, 10])
Clamped [0,5]: tensor([0, 0, 3, 5, 5])
```

#### `torch.sign(x)`

*Returns -1, 0, or 1 based on sign of each element.*

**Example:**

```python
import torch
x = torch.tensor([-3, 0, 5])
print(f"x: {x}")
print(f"sign(x): {torch.sign(x)}")
```

**Output:**

```
x: tensor([-3,  0,  5])
sign(x): tensor([-1,  0,  1])
```

#### `torch.sigmoid(x)`

*Sigmoid function: 1/(1+e^-x). Maps values to (0, 1). Used in binary classification.*

**Example:**

```python
import torch
x = torch.tensor([-2.0, 0.0, 2.0])
print(f"x: {x}")
print(f"sigmoid(x): {torch.sigmoid(x)}")
```

**Output:**

```
x: tensor([-2.,  0.,  2.])
sigmoid(x): tensor([0.1192, 0.5000, 0.8808])
```

---

## ≡ Linear Algebra (torch.*)

#### `torch.mm(a, b)  # 2D matrix mult`

*Matrix multiplication for 2D tensors. Use matmul for batched or higher dims.*

**Example:**

```python
import torch
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
result = torch.mm(a, b)
print("A @ B =")
print(result)
```

**Output:**

```
A @ B =
tensor([[19, 22],
        [43, 50]])
```

#### `torch.bmm(a, b)  # batch mm`

*Batched matrix multiplication. First dim is batch size.*

**Example:**

```python
import torch
a = torch.randn(10, 3, 4)  # batch of 10 matrices
b = torch.randn(10, 4, 5)
result = torch.bmm(a, b)
print(f"Batch shapes: {a.shape} @ {b.shape} = {result.shape}")
```

**Output:**

```
Batch shapes: torch.Size([10, 3, 4]) @ torch.Size([10, 4, 5]) = torch.Size([10, 3, 5])
```

#### `torch.mv(mat, vec)  # matrix-vector`

*Matrix-vector multiplication. vec treated as column vector.*

**Example:**

```python
import torch
mat = torch.tensor([[1, 2], [3, 4]])
vec = torch.tensor([1, 1])
result = torch.mv(mat, vec)
print(f"Matrix @ vector = {result}")
```

**Output:**

```
Matrix @ vector = tensor([3, 7])
```

#### `torch.dot(a, b)  # 1D dot product`

*Dot product of 1D tensors. Sum of element-wise products.*

**Example:**

```python
import torch
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
result = torch.dot(a, b)
print(f"{a} . {b} = {result}")
```

**Output:**

```
tensor([1., 2., 3.]) . tensor([4., 5., 6.]) = 32.0
```

#### `torch.det(x)  # determinant`

*Computes determinant of square matrix.*

**Example:**

```python
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
det = torch.det(x)
print("Matrix:")
print(x)
print(f"Determinant: {det:.4f}")
```

**Output:**

```
Matrix:
tensor([[1., 2.],
        [3., 4.]])
Determinant: -2.0000
```

#### `torch.inverse(x)  # matrix inverse`

*Computes matrix inverse. A @ A^-1 = Identity.*

**Example:**

```python
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
inv = torch.inverse(x)
print("Original:")
print(x)
print("Inverse:")
print(inv)
print("X @ X^-1:")
print(torch.mm(x, inv))
```

**Output:**

```
Original:
tensor([[1., 2.],
        [3., 4.]])
Inverse:
tensor([[-2.0000,  1.0000],
        [ 1.5000, -0.5000]])
X @ X^-1:
tensor([[1., 0.],
        [0., 1.]])
```

#### `torch.svd(x)  # singular value decomp`

*Singular Value Decomposition. X = U @ diag(S) @ V^T*

**Example:**

```python
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
U, S, V = torch.svd(x)
print(f"Original shape: {x.shape}")
print(f"U: {U.shape}, S: {S.shape}, V: {V.shape}")
print(f"Singular values: {S}")
```

**Output:**

```
Original shape: torch.Size([3, 2])
U: torch.Size([3, 2]), S: torch.Size([2]), V: torch.Size([2, 2])
Singular values: tensor([9.5255, 0.5143])
```

#### `torch.eig(x)  # eigenvalues`

*Computes eigenvalues. Use linalg.eig for general, linalg.eigvalsh for symmetric.*

**Example:**

```python
import torch
x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
eigenvalues = torch.linalg.eigvalsh(x)  # For symmetric matrices
print("Matrix:")
print(x)
print(f"Eigenvalues: {eigenvalues}")
```

**Output:**

```
Matrix:
tensor([[1., 2.],
        [2., 1.]])
Eigenvalues: tensor([-1.,  3.])
```

#### `torch.linalg.norm(x, ord)`

*Computes matrix or vector norm. Frobenius is default for matrices.*

**Example:**

```python
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("Matrix:")
print(x)
print(f"Frobenius norm: {torch.linalg.norm(x):.4f}")
print(f"L1 norm: {torch.linalg.norm(x, ord=1):.4f}")
```

**Output:**

```
Matrix:
tensor([[1., 2.],
        [3., 4.]])
Frobenius norm: 5.4772
L1 norm: 6.0000
```

#### `torch.linalg.solve(A, b)`

*Solves linear system Ax = b. More stable than computing inverse.*

**Example:**

```python
import torch
A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
b = torch.tensor([9.0, 8.0])
x = torch.linalg.solve(A, b)
print(f"Solving Ax = b")
print(f"x = {x}")
print(f"Verify A@x = {A @ x}")
```

**Output:**

```
Solving Ax = b
x = tensor([2., 3.])
Verify A@x = tensor([9., 8.])
```

#### `torch.trace(x)  # sum of diagonal`

*Sum of diagonal elements. Trace of identity is its dimension.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix:")
print(x)
print(f"Trace: {torch.trace(x)}")  # 1+5+9 = 15
```

**Output:**

```
Matrix:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
Trace: 15
```

#### `torch.outer(a, b)  # outer product`

*Outer product of two vectors. Result shape is (len(a), len(b)).*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5])
result = torch.outer(a, b)
print(f"a: {a}, b: {b}")
print("Outer product:")
print(result)
```

**Output:**

```
a: tensor([1, 2, 3]), b: tensor([4, 5])
Outer product:
tensor([[ 4,  5],
        [ 8, 10],
        [12, 15]])
```

---

## ◈ Neural Network (F.*)

### Activations (non-linear):

#### `F.relu(x)  # max(0, x)`

*Rectified Linear Unit. Returns max(0, x). Most common activation.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {x}")
print(f"ReLU:  {F.relu(x)}")
```

**Output:**

```
Input: tensor([-2., -1.,  0.,  1.,  2.])
ReLU:  tensor([0., 0., 0., 1., 2.])
```

#### `F.leaky_relu(x, neg_slope)`

*Like ReLU but allows small negative values. Prevents 'dying ReLU' problem.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {x}")
print(f"Leaky ReLU: {F.leaky_relu(x, 0.1)}")
```

**Output:**

```
Input: tensor([-2., -1.,  0.,  1.,  2.])
Leaky ReLU: tensor([-0.2000, -0.1000,  0.0000,  1.0000,  2.0000])
```

#### `F.gelu(x)  # Gaussian Error`

*Gaussian Error Linear Unit. Used in transformers (BERT, GPT).*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {x}")
print(f"GELU:  {F.gelu(x)}")
```

**Output:**

```
Input: tensor([-2., -1.,  0.,  1.,  2.])
GELU:  tensor([-0.0455, -0.1587,  0.0000,  0.8413,  1.9545])
```

#### `F.sigmoid(x)  # 1/(1+e^-x)`

*Maps values to (0, 1). Used for binary classification output.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.tensor([-2.0, 0.0, 2.0])
print(f"Input: {x}")
print(f"Sigmoid: {F.sigmoid(x)}")
```

**Output:**

```
Input: tensor([-2.,  0.,  2.])
Sigmoid: tensor([0.1192, 0.5000, 0.8808])
```

#### `F.tanh(x)  # hyperbolic tan`

*Maps values to (-1, 1). Zero-centered, better than sigmoid for hidden layers.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.tensor([-2.0, 0.0, 2.0])
print(f"Input: {x}")
print(f"Tanh: {F.tanh(x)}")
```

**Output:**

```
Input: tensor([-2.,  0.,  2.])
Tanh: tensor([-0.9640,  0.0000,  0.9640])
```

#### `F.softmax(x, dim)  # probabilities`

*Converts logits to probabilities (sum to 1). Used for multi-class output.*

**Example:**

```python
import torch
import torch.nn.functional as F
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(f"Logits: {logits}")
print(f"Softmax: {probs}")
print(f"Sum: {probs.sum():.4f}")
```

**Output:**

```
Logits: tensor([2.0000, 1.0000, 0.1000])
Softmax: tensor([0.6590, 0.2424, 0.0986])
Sum: 1.0000
```

#### `F.log_softmax(x, dim)`

*Log of softmax. More numerically stable. Used with NLLLoss.*

**Example:**

```python
import torch
import torch.nn.functional as F
logits = torch.tensor([2.0, 1.0, 0.1])
log_probs = F.log_softmax(logits, dim=0)
print(f"Logits: {logits}")
print(f"Log softmax: {log_probs}")
```

**Output:**

```
Logits: tensor([2.0000, 1.0000, 0.1000])
Log softmax: tensor([-0.4170, -1.4170, -2.3170])
```

### Regularization:

#### `F.dropout(x, p, training)`

*Randomly zeros elements with probability p. Only active during training.*

**Example:**

```python
import torch
import torch.nn.functional as F
torch.manual_seed(42)
x = torch.ones(10)
dropped = F.dropout(x, p=0.5, training=True)
print(f"Original: {x}")
print(f"Dropout (p=0.5): {dropped}")
```

**Output:**

```
Original: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
Dropout (p=0.5): tensor([2., 2., 2., 2., 0., 2., 0., 0., 2., 2.])
```

#### `F.batch_norm(x, ...)  # normalize`

*Normalizes across batch dimension. Stabilizes training.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.randn(2, 3, 4, 4)  # batch, channels, H, W
mean = torch.zeros(3)
var = torch.ones(3)
result = F.batch_norm(x, mean, var)
print(f"Input shape: {x.shape}")
print(f"After batch_norm: mean~{result.mean():.4f}, std~{result.std():.4f}")
```

**Output:**

```
Input shape: torch.Size([2, 3, 4, 4])
After batch_norm: mean~0.1266, std~0.9259
```

#### `F.layer_norm(x, shape)`

*Normalizes across specified dimensions. Used in transformers.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.randn(2, 3, 4)
result = F.layer_norm(x, [3, 4])
print(f"Input shape: {x.shape}")
print(f"After layer_norm: mean~{result.mean():.4f}, std~{result.std():.4f}")
```

**Output:**

```
Input shape: torch.Size([2, 3, 4])
After layer_norm: mean~0.0000, std~1.0215
```

---

## × Loss Functions (F.*)

#### `F.mse_loss(pred, target)`

*Mean Squared Error. Average of squared differences. For regression.*

**Example:**

```python
import torch
import torch.nn.functional as F
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.0, 2.5])
loss = F.mse_loss(pred, target)
print(f"Prediction: {pred}")
print(f"Target: {target}")
print(f"MSE Loss: {loss:.4f}")
```

**Output:**

```
Prediction: tensor([1., 2., 3.])
Target: tensor([1.5000, 2.0000, 2.5000])
MSE Loss: 0.1667
```

#### `F.l1_loss(pred, target)`

*Mean Absolute Error. Less sensitive to outliers than MSE.*

**Example:**

```python
import torch
import torch.nn.functional as F
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.0, 2.5])
loss = F.l1_loss(pred, target)
print(f"Prediction: {pred}")
print(f"Target: {target}")
print(f"L1 Loss: {loss:.4f}")
```

**Output:**

```
Prediction: tensor([1., 2., 3.])
Target: tensor([1.5000, 2.0000, 2.5000])
L1 Loss: 0.3333
```

#### `F.cross_entropy(logits, labels)`

*Combines log_softmax and NLLLoss. Standard for multi-class classification.*

**Example:**

```python
import torch
import torch.nn.functional as F
logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 2.0, 0.5]])
labels = torch.tensor([0, 1])  # Class indices
loss = F.cross_entropy(logits, labels)
print(f"Logits: {logits}")
print(f"Labels: {labels}")
print(f"Cross Entropy Loss: {loss:.4f}")
```

**Output:**

```
Logits: tensor([[2.0000, 0.5000, 0.1000],
        [0.1000, 2.0000, 0.5000]])
Labels: tensor([0, 1])
Cross Entropy Loss: 0.3168
```

#### `F.nll_loss(log_probs, labels)`

*Negative Log Likelihood. Use with log_softmax output.*

**Example:**

```python
import torch
import torch.nn.functional as F
log_probs = F.log_softmax(torch.tensor([[2.0, 0.5], [0.5, 2.0]]), dim=1)
labels = torch.tensor([0, 1])
loss = F.nll_loss(log_probs, labels)
print(f"NLL Loss: {loss:.4f}")
```

**Output:**

```
NLL Loss: 0.2014
```

#### `F.binary_cross_entropy(pred, target)`

*Binary Cross Entropy. For binary classification with sigmoid output.*

**Example:**

```python
import torch
import torch.nn.functional as F
pred = torch.tensor([0.8, 0.4, 0.9])  # Probabilities
target = torch.tensor([1.0, 0.0, 1.0])  # Binary labels
loss = F.binary_cross_entropy(pred, target)
print(f"Pred: {pred}")
print(f"Target: {target}")
print(f"BCE Loss: {loss:.4f}")
```

**Output:**

```
Pred: tensor([0.8000, 0.4000, 0.9000])
Target: tensor([1., 0., 1.])
BCE Loss: 0.2798
```

#### `F.kl_div(log_pred, target)`

*Kullback-Leibler Divergence. Measures difference between distributions.*

**Example:**

```python
import torch
import torch.nn.functional as F
log_pred = F.log_softmax(torch.tensor([0.5, 0.3, 0.2]), dim=0)
target = F.softmax(torch.tensor([0.4, 0.4, 0.2]), dim=0)
loss = F.kl_div(log_pred, target, reduction='sum')
print(f"KL Divergence: {loss:.4f}")
```

**Output:**

```
KL Divergence: 0.0035
```

#### `F.cosine_similarity(x1, x2, dim)`

*Measures angle between vectors. 1 = same direction, -1 = opposite.*

**Example:**

```python
import torch
import torch.nn.functional as F
x1 = torch.tensor([[1.0, 0.0, 0.0]])
x2 = torch.tensor([[1.0, 1.0, 0.0]])
sim = F.cosine_similarity(x1, x2)
print(f"x1: {x1}")
print(f"x2: {x2}")
print(f"Cosine similarity: {sim}")
```

**Output:**

```
x1: tensor([[1., 0., 0.]])
x2: tensor([[1., 1., 0.]])
Cosine similarity: tensor([0.7071])
```

#### `F.triplet_margin_loss(...)`

*Learns embeddings where similar items are close, different are far.*

**Example:**

```python
import torch
import torch.nn.functional as F
anchor = torch.randn(3, 128)
positive = anchor + 0.1 * torch.randn(3, 128)
negative = torch.randn(3, 128)
loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
print(f"Triplet loss: {loss:.4f}")
```

**Output:**

```
Triplet loss: 0.0000
```

---

## ▣ Pooling & Conv (F.*)

#### `F.max_pool2d(x, kernel_size)`

*Downsamples by taking max in each window. Reduces spatial dimensions.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.arange(16).float().view(1, 1, 4, 4)
print("Input (4x4):")
print(x.squeeze())
result = F.max_pool2d(x, kernel_size=2)
print("MaxPool2d (kernel=2):")
print(result.squeeze())
```

**Output:**

```
Input (4x4):
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]])
MaxPool2d (kernel=2):
tensor([[ 5.,  7.],
        [13., 15.]])
```

#### `F.avg_pool2d(x, kernel_size)`

*Downsamples by averaging each window. Smoother than max pooling.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.arange(16).float().view(1, 1, 4, 4)
print("Input (4x4):")
print(x.squeeze())
result = F.avg_pool2d(x, kernel_size=2)
print("AvgPool2d (kernel=2):")
print(result.squeeze())
```

**Output:**

```
Input (4x4):
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]])
AvgPool2d (kernel=2):
tensor([[ 2.5000,  4.5000],
        [10.5000, 12.5000]])
```

#### `F.adaptive_max_pool2d(x, output_size)`

*Pools to exact output size regardless of input size.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.randn(1, 1, 7, 7)
result = F.adaptive_max_pool2d(x, output_size=(2, 2))
print(f"Input: {x.shape} -> Output: {result.shape}")
```

**Output:**

```
Input: torch.Size([1, 1, 7, 7]) -> Output: torch.Size([1, 1, 2, 2])
```

#### `F.conv2d(x, weight, bias)`

*2D convolution. Core operation for CNNs. Extracts spatial features.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.randn(1, 1, 5, 5)  # batch, in_ch, H, W
weight = torch.randn(1, 1, 3, 3)  # out_ch, in_ch, kH, kW
result = F.conv2d(x, weight)
print(f"Input: {x.shape}")
print(f"Weight: {weight.shape}")
print(f"Output: {result.shape}")
```

**Output:**

```
Input: torch.Size([1, 1, 5, 5])
Weight: torch.Size([1, 1, 3, 3])
Output: torch.Size([1, 1, 3, 3])
```

#### `F.conv_transpose2d(x, weight)`

*Transposed convolution (deconvolution). Upsamples spatial dimensions.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.randn(1, 1, 3, 3)
weight = torch.randn(1, 1, 3, 3)
result = F.conv_transpose2d(x, weight)
print(f"Input: {x.shape}")
print(f"Output: {result.shape}")
```

**Output:**

```
Input: torch.Size([1, 1, 3, 3])
Output: torch.Size([1, 1, 5, 5])
```

#### `F.interpolate(x, size, mode)`

*Resizes tensor using interpolation. Modes: nearest, bilinear, bicubic.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.randn(1, 1, 4, 4)
result = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)
print(f"Input: {x.shape} -> Output: {result.shape}")
```

**Output:**

```
Input: torch.Size([1, 1, 4, 4]) -> Output: torch.Size([1, 1, 8, 8])
```

#### `F.pad(x, pad, mode)`

*Adds padding around tensor. Used in convolutions. Modes: constant, reflect, replicate.*

**Example:**

```python
import torch
import torch.nn.functional as F
x = torch.arange(4).float().view(1, 1, 2, 2)
print("Input:")
print(x.squeeze())
padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
print("Padded (1 on each side):")
print(padded.squeeze())
```

**Output:**

```
Input:
tensor([[0., 1.],
        [2., 3.]])
Padded (1 on each side):
tensor([[0., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 2., 3., 0.],
        [0., 0., 0., 0.]])
```

---

## ∞ Advanced Ops (torch.*)

#### `torch.einsum('ij,jk->ik', a, b)`

*Einstein summation. Flexible notation for many tensor operations.*

**Example:**

```python
import torch
a = torch.randn(2, 3)
b = torch.randn(3, 4)
result = torch.einsum('ij,jk->ik', a, b)
print(f"einsum('ij,jk->ik'): {a.shape} x {b.shape} = {result.shape}")
# Verify it's matrix multiplication
print(f"Same as matmul: {torch.allclose(result, a @ b)}")
```

**Output:**

```
einsum('ij,jk->ik'): torch.Size([2, 3]) x torch.Size([3, 4]) = torch.Size([2, 4])
Same as matmul: True
```

#### `torch.topk(x, k, dim)  # top k values`

*Returns k largest values and their indices. Faster than full sort.*

**Example:**

```python
import torch
x = torch.tensor([1, 5, 3, 9, 2, 7])
vals, idxs = torch.topk(x, k=3)
print(f"Input: {x}")
print(f"Top 3 values: {vals}")
print(f"Top 3 indices: {idxs}")
```

**Output:**

```
Input: tensor([1, 5, 3, 9, 2, 7])
Top 3 values: tensor([9, 7, 5])
Top 3 indices: tensor([3, 5, 1])
```

#### `torch.sort(x, dim)  # sorted values`

*Sorts tensor along dimension. Returns values and original indices.*

**Example:**

```python
import torch
x = torch.tensor([3, 1, 4, 1, 5, 9])
vals, idxs = torch.sort(x)
print(f"Original: {x}")
print(f"Sorted: {vals}")
print(f"Indices: {idxs}")
```

**Output:**

```
Original: tensor([3, 1, 4, 1, 5, 9])
Sorted: tensor([1, 1, 3, 4, 5, 9])
Indices: tensor([1, 3, 0, 2, 4, 5])
```

#### `torch.argsort(x, dim)  # sort indices`

*Returns indices that would sort the tensor.*

**Example:**

```python
import torch
x = torch.tensor([3, 1, 4, 1, 5])
idxs = torch.argsort(x)
print(f"Original: {x}")
print(f"Argsort: {idxs}")
print(f"Sorted via indices: {x[idxs]}")
```

**Output:**

```
Original: tensor([3, 1, 4, 1, 5])
Argsort: tensor([1, 3, 0, 2, 4])
Sorted via indices: tensor([1, 1, 3, 4, 5])
```

#### `torch.unique(x)  # unique values`

*Returns unique elements. Optional return_counts for frequency.*

**Example:**

```python
import torch
x = torch.tensor([1, 2, 2, 3, 1, 3, 3, 4])
unique = torch.unique(x)
print(f"Original: {x}")
print(f"Unique: {unique}")
```

**Output:**

```
Original: tensor([1, 2, 2, 3, 1, 3, 3, 4])
Unique: tensor([1, 2, 3, 4])
```

#### `torch.cat([t1, t2], dim)  # concat`

*Concatenates tensors along existing dimension. Shapes must match except in dim.*

**Example:**

```python
import torch
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6]])
result = torch.cat([a, b], dim=0)
print("Concatenated along dim 0:")
print(result)
```

**Output:**

```
Concatenated along dim 0:
tensor([[1, 2],
        [3, 4],
        [5, 6]])
```

#### `torch.stack([t1, t2], dim)  # new dim`

*Stacks tensors along NEW dimension. All tensors must have same shape.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
result = torch.stack([a, b], dim=0)
print("Stacked (creates new dim):")
print(result)
print(f"Shape: {result.shape}")
```

**Output:**

```
Stacked (creates new dim):
tensor([[1, 2, 3],
        [4, 5, 6]])
Shape: torch.Size([2, 3])
```

#### `torch.split(x, size, dim)  # split`

*Splits tensor into chunks of specified size. Last chunk may be smaller.*

**Example:**

```python
import torch
x = torch.arange(10)
splits = torch.split(x, 3)
print(f"Original: {x}")
print("Split into chunks of 3:")
for i, s in enumerate(splits):
    print(f"  {i}: {s}")
```

**Output:**

```
Original: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Split into chunks of 3:
  0: tensor([0, 1, 2])
  1: tensor([3, 4, 5])
  2: tensor([6, 7, 8])
  3: tensor([9])
```

#### `torch.chunk(x, chunks, dim)  # chunks`

*Splits tensor into specified number of chunks.*

**Example:**

```python
import torch
x = torch.arange(12)
chunks = torch.chunk(x, 4)
print(f"Original: {x}")
print("Split into 4 chunks:")
for i, c in enumerate(chunks):
    print(f"  {i}: {c}")
```

**Output:**

```
Original: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
Split into 4 chunks:
  0: tensor([0, 1, 2])
  1: tensor([3, 4, 5])
  2: tensor([6, 7, 8])
  3: tensor([ 9, 10, 11])
```

#### `torch.broadcast_to(x, shape)`

*Explicitly broadcasts tensor to new shape. No data copy.*

**Example:**

```python
import torch
x = torch.tensor([1, 2, 3])
result = torch.broadcast_to(x, (3, 3))
print(f"Original: {x}")
print("Broadcast to (3, 3):")
print(result)
```

**Output:**

```
Original: tensor([1, 2, 3])
Broadcast to (3, 3):
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
```

#### `torch.flatten(x, start, end)`

*Flattens tensor. Optional start/end dims for partial flattening.*

**Example:**

```python
import torch
x = torch.randn(2, 3, 4)
flat = torch.flatten(x)
partial = torch.flatten(x, start_dim=1)
print(f"Original: {x.shape}")
print(f"Fully flat: {flat.shape}")
print(f"Flatten from dim 1: {partial.shape}")
```

**Output:**

```
Original: torch.Size([2, 3, 4])
Fully flat: torch.Size([24])
Flatten from dim 1: torch.Size([2, 12])
```

---

## ∂ Autograd (mixed)

#### `x.requires_grad_(True)`

*Enables gradient tracking for tensor. Required for backpropagation.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Before: requires_grad = {x.requires_grad}")
x.requires_grad_(True)
print(f"After: requires_grad = {x.requires_grad}")
```

**Output:**

```
Before: requires_grad = False
After: requires_grad = True
```

#### `y.backward()`

*Computes gradients via backpropagation. Stores in .grad attribute.*

**Example:**

```python
import torch
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()  # y = x1^2 + x2^2
y.backward()
print(f"x = {x}")
print(f"y = x^2.sum() = {y}")
print(f"dy/dx = {x.grad}")  # 2*x
```

**Output:**

```
x = tensor([2., 3.], requires_grad=True)
y = x^2.sum() = 13.0
dy/dx = tensor([4., 6.])
```

#### `x.grad  # gradient`

*Stores accumulated gradients after backward(). Reset with zero_grad().*

**Example:**

```python
import torch
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"x = {x}")
print(f"y = x^2 = {y}")
print(f"dy/dx = {x.grad}")
```

**Output:**

```
x = 3.0
y = x^2 = 9.0
dy/dx = 6.0
```

#### `x.detach()`

*Returns tensor detached from computation graph. Stops gradient flow.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = y.detach()
print(f"y requires_grad: {y.requires_grad}")
print(f"z requires_grad: {z.requires_grad}")
```

**Output:**

```
y requires_grad: True
z requires_grad: False
```

#### `x.clone()`

*Creates copy with same data. Modifications don't affect original.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0])
y = x.clone()
y[0] = 99
print(f"Original x: {x}")
print(f"Cloned y: {y}")
```

**Output:**

```
Original x: tensor([1., 2.])
Cloned y: tensor([99.,  2.])
```

#### `with torch.no_grad():`

*Context manager that disables gradient computation. For inference.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
with torch.no_grad():
    y = x * 2
    print(f"Inside no_grad: y.requires_grad = {y.requires_grad}")
z = x * 2
print(f"Outside: z.requires_grad = {z.requires_grad}")
```

**Output:**

```
Inside no_grad: y.requires_grad = False
Outside: z.requires_grad = True
```

#### `torch.autograd.grad(y, x)`

*Computes gradient without modifying .grad. Useful for second derivatives.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 3).sum()  # y = x1^3 + x2^3
grad = torch.autograd.grad(y, x)
print(f"x = {x}")
print(f"dy/dx = {grad[0]}")  # 3*x^2
```

**Output:**

```
x = tensor([1., 2.], requires_grad=True)
dy/dx = tensor([ 3., 12.])
```

#### `optimizer.zero_grad()`

*Resets gradients to zero. Call before each backward pass.*

**Example:**

```python
import torch
import torch.optim as optim
x = torch.tensor([1.0], requires_grad=True)
optimizer = optim.SGD([x], lr=0.1)
y = x ** 2
y.backward()
print(f"Grad before zero_grad: {x.grad}")
optimizer.zero_grad()
print(f"Grad after zero_grad: {x.grad}")
```

**Output:**

```
Grad before zero_grad: tensor([2.])
Grad after zero_grad: None
```

#### `optimizer.step()`

*Updates parameters using computed gradients. x = x - lr * grad.*

**Example:**

```python
import torch
import torch.optim as optim
x = torch.tensor([5.0], requires_grad=True)
optimizer = optim.SGD([x], lr=0.1)
print(f"Before: x = {x.item():.4f}")
y = x ** 2
y.backward()
optimizer.step()
print(f"After step: x = {x.item():.4f}")
```

**Output:**

```
Before: x = 5.0000
After step: x = 4.0000
```

#### `torch.nn.utils.clip_grad_norm_(params, max)`

*Clips gradient norm to prevent exploding gradients. Essential for RNNs.*

**Example:**

```python
import torch
import torch.nn as nn
model = nn.Linear(10, 5)
x = torch.randn(1, 10)
loss = model(x).sum()
loss.backward()
norm_before = sum(p.grad.norm()**2 for p in model.parameters())**0.5
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
norm_after = sum(p.grad.norm()**2 for p in model.parameters())**0.5
print(f"Grad norm before: {norm_before:.4f}")
print(f"Grad norm after clip: {norm_after:.4f}")
```

**Output:**

```
Grad norm before: 6.8840
Grad norm after clip: 1.0000
```

---

## ◎ Device Ops (mixed)

#### `torch.cuda.is_available()`

*Checks if CUDA GPU is available. Use for device selection logic.*

**Example:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

**Output:**

```
CUDA available: False
MPS available: True
```

#### `torch.device('cuda'/'cpu'/'mps')`

*Creates device object. Use with .to(device) for device placement.*

**Example:**

```python
import torch
cpu = torch.device('cpu')
print(f"CPU device: {cpu}")
if torch.cuda.is_available():
    gpu = torch.device('cuda')
    print(f"GPU device: {gpu}")
elif torch.backends.mps.is_available():
    mps = torch.device('mps')
    print(f"MPS device: {mps}")
```

**Output:**

```
CPU device: cpu
MPS device: mps
```

#### `x.to(device)  # move to device`

*Moves tensor to specified device. Essential for GPU training.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0])
device = torch.device('cpu')
x_dev = x.to(device)
print(f"Tensor on: {x_dev.device}")
```

**Output:**

```
Tensor on: cpu
```

#### `x.to(dtype)  # change dtype`

*Converts tensor to different data type.*

**Example:**

```python
import torch
x = torch.tensor([1, 2, 3])
print(f"Original dtype: {x.dtype}")
x_float = x.to(torch.float32)
print(f"After to(float32): {x_float.dtype}")
```

**Output:**

```
Original dtype: torch.int64
After to(float32): torch.float32
```

#### `x.cuda(), x.cpu()  # shortcuts`

*Shortcut methods for moving between CPU and CUDA.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0])
print(f"Original device: {x.device}")
x_cpu = x.cpu()
print(f"After cpu(): {x_cpu.device}")
```

**Output:**

```
Original device: cpu
After cpu(): cpu
```

#### `x.device  # check current device`

*Returns device where tensor is stored.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0])
print(f"Device: {x.device}")
print(f"Device type: {x.device.type}")
```

**Output:**

```
Device: cpu
Device type: cpu
```

#### `x.is_cuda  # boolean check`

*Returns True if tensor is on CUDA device.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0])
print(f"is_cuda: {x.is_cuda}")
```

**Output:**

```
is_cuda: False
```

#### `torch.cuda.empty_cache()`

*Releases unused cached GPU memory. Helps with OOM errors.*

**Example:**

```python
import torch
# Free unused cached memory (only affects CUDA)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache cleared")
else:
    print("CUDA not available - cache clearing skipped")
```

**Output:**

```
CUDA not available - cache clearing skipped
```

#### `torch.cuda.device_count()`

*Returns number of available CUDA devices.*

**Example:**

```python
import torch
if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f"Number of GPUs: {count}")
else:
    print("CUDA not available")
```

**Output:**

```
CUDA not available
```

---

## ※ Utilities (mixed)

#### `x.dtype, x.shape, x.size()`

*Basic tensor properties. shape and size() are equivalent.*

**Example:**

```python
import torch
x = torch.randn(2, 3, 4)
print(f"dtype: {x.dtype}")
print(f"shape: {x.shape}")
print(f"size(): {x.size()}")
```

**Output:**

```
dtype: torch.float32
shape: torch.Size([2, 3, 4])
size(): torch.Size([2, 3, 4])
```

#### `x.numel()  # number of elements`

*Returns total number of elements. Product of all dimensions.*

**Example:**

```python
import torch
x = torch.randn(2, 3, 4)
print(f"Shape: {x.shape}")
print(f"Number of elements: {x.numel()}")
```

**Output:**

```
Shape: torch.Size([2, 3, 4])
Number of elements: 24
```

#### `x.dim()  # number of dimensions`

*Returns number of dimensions (rank) of tensor.*

**Example:**

```python
import torch
x = torch.randn(2, 3, 4)
print(f"Shape: {x.shape}")
print(f"Dimensions: {x.dim()}")
```

**Output:**

```
Shape: torch.Size([2, 3, 4])
Dimensions: 3
```

#### `x.ndimension()  # same as dim()`

*Alias for dim(). Returns number of dimensions.*

**Example:**

```python
import torch
x = torch.randn(2, 3, 4)
print(f"ndimension(): {x.ndimension()}")
print(f"Same as dim(): {x.dim()}")
```

**Output:**

```
ndimension(): 3
Same as dim(): 3
```

#### `x.is_contiguous()`

*Checks if tensor is contiguous in memory. Required for view().*

**Example:**

```python
import torch
x = torch.randn(3, 4)
y = x.transpose(0, 1)
print(f"Original is_contiguous: {x.is_contiguous()}")
print(f"Transposed is_contiguous: {y.is_contiguous()}")
```

**Output:**

```
Original is_contiguous: True
Transposed is_contiguous: False
```

#### `x.float(), x.int(), x.long()`

*Shortcut methods for dtype conversion.*

**Example:**

```python
import torch
x = torch.tensor([1, 2, 3])
print(f"Original: {x.dtype}")
print(f"float(): {x.float().dtype}")
print(f"long(): {x.long().dtype}")
```

**Output:**

```
Original: torch.int64
float(): torch.float32
long(): torch.int64
```

#### `x.half(), x.double()  # fp16, fp64`

*Half precision (16-bit) for faster training, double for higher precision.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0])
print(f"Original: {x.dtype}")
print(f"half() (fp16): {x.half().dtype}")
print(f"double() (fp64): {x.double().dtype}")
```

**Output:**

```
Original: torch.float32
half() (fp16): torch.float16
double() (fp64): torch.float64
```

#### `torch.from_numpy(arr)`

*Converts NumPy array to tensor. Shares memory with original.*

**Example:**

```python
import torch
import numpy as np
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)
print(f"NumPy: {arr}, dtype={arr.dtype}")
print(f"Tensor: {x}, dtype={x.dtype}")
```

**Output:**

```
NumPy: [1 2 3], dtype=int64
Tensor: tensor([1, 2, 3]), dtype=torch.int64
```

#### `x.numpy()  # CPU only`

*Converts tensor to NumPy array. Must be on CPU.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
arr = x.numpy()
print(f"Tensor: {x}")
print(f"NumPy: {arr}, dtype={arr.dtype}")
```

**Output:**

```
Tensor: tensor([1., 2., 3.])
NumPy: [1. 2. 3.], dtype=float32
```

#### `torch.save(obj, path)`

*Saves tensor or model to file. Uses pickle serialization.*

**Example:**

```python
import torch
import os
x = torch.tensor([1, 2, 3])
torch.save(x, '/tmp/tensor.pt')
print(f"Saved tensor to /tmp/tensor.pt")
print(f"File size: {os.path.getsize('/tmp/tensor.pt')} bytes")
```

**Output:**

```
Saved tensor to /tmp/tensor.pt
File size: 1570 bytes
```

#### `torch.load(path)`

*Loads saved tensor or model from file.*

**Example:**

```python
import torch
torch.save(torch.tensor([1, 2, 3]), '/tmp/tensor.pt')
loaded = torch.load('/tmp/tensor.pt', weights_only=True)
print(f"Loaded: {loaded}")
```

**Output:**

```
Loaded: tensor([1, 2, 3])
```

#### `torch.manual_seed(seed)`

*Sets random seed for reproducibility.*

**Example:**

```python
import torch
torch.manual_seed(42)
a = torch.rand(3)
torch.manual_seed(42)
b = torch.rand(3)
print(f"First: {a}")
print(f"Second (same seed): {b}")
print(f"Equal: {torch.equal(a, b)}")
```

**Output:**

```
First: tensor([0.8823, 0.9150, 0.3829])
Second (same seed): tensor([0.8823, 0.9150, 0.3829])
Equal: True
```

---

## ≈ Comparison (torch.*)

#### `torch.eq(a, b) or a == b`

*Element-wise equality comparison. Returns boolean tensor.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 0, 3])
print(f"a: {a}")
print(f"b: {b}")
print(f"a == b: {a == b}")
```

**Output:**

```
a: tensor([1, 2, 3])
b: tensor([1, 0, 3])
a == b: tensor([ True, False,  True])
```

#### `torch.ne(a, b) or a != b`

*Element-wise not-equal comparison.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 0, 3])
print(f"a != b: {a != b}")
```

**Output:**

```
a != b: tensor([False,  True, False])
```

#### `torch.gt(a, b) or a > b`

*Element-wise greater-than comparison.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])
print(f"a: {a}, b: {b}")
print(f"a > b: {a > b}")
```

**Output:**

```
a: tensor([1, 2, 3]), b: tensor([2, 2, 2])
a > b: tensor([False, False,  True])
```

#### `torch.lt(a, b) or a < b`

*Element-wise less-than comparison.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])
print(f"a < b: {a < b}")
```

**Output:**

```
a < b: tensor([ True, False, False])
```

#### `torch.ge(a, b) or a >= b`

*Element-wise greater-than-or-equal comparison.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])
print(f"a >= b: {a >= b}")
```

**Output:**

```
a >= b: tensor([False,  True,  True])
```

#### `torch.le(a, b) or a <= b`

*Element-wise less-than-or-equal comparison.*

**Example:**

```python
import torch
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])
print(f"a <= b: {a <= b}")
```

**Output:**

```
a <= b: tensor([ True,  True, False])
```

#### `torch.allclose(a, b, rtol, atol)`

*Checks if all elements are close within tolerance. For float comparison.*

**Example:**

```python
import torch
a = torch.tensor([1.0, 2.0])
b = torch.tensor([1.0001, 2.0001])
print(f"a: {a}")
print(f"b: {b}")
print(f"allclose (default tol): {torch.allclose(a, b)}")
print(f"allclose (rtol=1e-3): {torch.allclose(a, b, rtol=1e-3)}")
```

**Output:**

```
a: tensor([1., 2.])
b: tensor([1.0001, 2.0001])
allclose (default tol): False
allclose (rtol=1e-3): True
```

#### `torch.isnan(x)`

*Returns True where elements are NaN (Not a Number).*

**Example:**

```python
import torch
x = torch.tensor([1.0, float('nan'), 3.0])
print(f"x: {x}")
print(f"isnan: {torch.isnan(x)}")
```

**Output:**

```
x: tensor([1., nan, 3.])
isnan: tensor([False,  True, False])
```

#### `torch.isinf(x)`

*Returns True where elements are infinite.*

**Example:**

```python
import torch
x = torch.tensor([1.0, float('inf'), float('-inf')])
print(f"x: {x}")
print(f"isinf: {torch.isinf(x)}")
```

**Output:**

```
x: tensor([1., inf, -inf])
isinf: tensor([False,  True,  True])
```

#### `torch.isfinite(x)`

*Returns True where elements are finite (not inf, not nan).*

**Example:**

```python
import torch
x = torch.tensor([1.0, float('inf'), float('nan')])
print(f"x: {x}")
print(f"isfinite: {torch.isfinite(x)}")
```

**Output:**

```
x: tensor([1., inf, nan])
isfinite: tensor([ True, False, False])
```

---

## ● Tensor Methods (x.*)

#### `x.T  # transpose (2D)`

*Shorthand for 2D transpose. For higher dims, use permute.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Original:")
print(x)
print("x.T:")
print(x.T)
```

**Output:**

```
Original:
tensor([[1, 2, 3],
        [4, 5, 6]])
x.T:
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

#### `x.H  # conjugate transpose`

*Hermitian transpose for complex tensors. Transpose + conjugate.*

**Example:**

```python
import torch
x = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]])
print("Original:")
print(x)
print("x.H (conjugate transpose):")
print(x.H)
```

**Output:**

```
Original:
tensor([[1.+2.j, 3.+4.j],
        [5.+6.j, 7.+8.j]])
x.H (conjugate transpose):
tensor([[1.-2.j, 5.-6.j],
        [3.-4.j, 7.-8.j]])
```

#### `x.real, x.imag  # complex parts`

*Access real and imaginary parts of complex tensor.*

**Example:**

```python
import torch
x = torch.tensor([1+2j, 3+4j])
print(f"Complex: {x}")
print(f"Real: {x.real}")
print(f"Imag: {x.imag}")
```

**Output:**

```
Complex: tensor([1.+2.j, 3.+4.j])
Real: tensor([1., 3.])
Imag: tensor([2., 4.])
```

#### `x.abs(), x.neg()  # absolute, negate`

*Methods for absolute value and negation.*

**Example:**

```python
import torch
x = torch.tensor([-3, -1, 0, 2, 4])
print(f"x: {x}")
print(f"abs: {x.abs()}")
print(f"neg: {x.neg()}")
```

**Output:**

```
x: tensor([-3, -1,  0,  2,  4])
abs: tensor([3, 1, 0, 2, 4])
neg: tensor([ 3,  1,  0, -2, -4])
```

#### `x.reciprocal(), x.pow(n)`

*Reciprocal (1/x) and power operations as methods.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 4.0])
print(f"x: {x}")
print(f"reciprocal: {x.reciprocal()}")
print(f"pow(2): {x.pow(2)}")
```

**Output:**

```
x: tensor([1., 2., 4.])
reciprocal: tensor([1.0000, 0.5000, 0.2500])
pow(2): tensor([ 1.,  4., 16.])
```

#### `x.sqrt(), x.exp(), x.log()`

*Common math operations as tensor methods.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 4.0, 9.0])
print(f"x: {x}")
print(f"sqrt: {x.sqrt()}")
print(f"exp: {torch.tensor([0.0, 1.0]).exp()}")
print(f"log: {torch.tensor([1.0, 2.718]).log()}")
```

**Output:**

```
x: tensor([1., 4., 9.])
sqrt: tensor([1., 2., 3.])
exp: tensor([1.0000, 2.7183])
log: tensor([0.0000, 0.9999])
```

#### `x.item()  # get scalar value`

*Extracts scalar value from single-element tensor as Python number.*

**Example:**

```python
import torch
x = torch.tensor(3.14159)
val = x.item()
print(f"Tensor: {x}")
print(f"Python float: {val}")
print(f"Type: {type(val)}")
```

**Output:**

```
Tensor: 3.141590118408203
Python float: 3.141590118408203
Type: <class 'float'>
```

#### `x.tolist()  # to Python list`

*Converts tensor to nested Python list.*

**Example:**

```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
lst = x.tolist()
print(f"Tensor: {x}")
print(f"List: {lst}")
print(f"Type: {type(lst)}")
```

**Output:**

```
Tensor: tensor([[1, 2],
        [3, 4]])
List: [[1, 2], [3, 4]]
Type: <class 'list'>
```

#### `x.all(), x.any()  # boolean checks`

*Checks if all or any elements are True.*

**Example:**

```python
import torch
x = torch.tensor([True, True, False])
print(f"x: {x}")
print(f"all: {x.all()}")
print(f"any: {x.any()}")
```

**Output:**

```
x: tensor([ True,  True, False])
all: False
any: True
```

#### `x.nonzero()  # non-zero indices`

*Returns indices of non-zero elements.*

**Example:**

```python
import torch
x = torch.tensor([0, 1, 0, 2, 0, 3])
indices = x.nonzero()
print(f"x: {x}")
print(f"Non-zero indices: {indices.squeeze()}")
```

**Output:**

```
x: tensor([0, 1, 0, 2, 0, 3])
Non-zero indices: tensor([1, 3, 5])
```

#### `x.fill_(val), x.zero_()  # in-place`

*In-place fill with value or zeros. Underscore suffix = in-place.*

**Example:**

```python
import torch
x = torch.empty(2, 3)
x.fill_(5.0)
print("After fill_(5.0):")
print(x)
x.zero_()
print("After zero_():")
print(x)
```

**Output:**

```
After fill_(5.0):
tensor([[5., 5., 5.],
        [5., 5., 5.]])
After zero_():
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

#### `x.normal_(), x.uniform_()  # random`

*In-place random initialization. Useful for weight init.*

**Example:**

```python
import torch
torch.manual_seed(42)
x = torch.empty(2, 3)
x.normal_(mean=0, std=1)
print("After normal_(0, 1):")
print(x)
x.uniform_(0, 1)
print("After uniform_(0, 1):")
print(x)
```

**Output:**

```
After normal_(0, 1):
tensor([[ 0.3367,  0.1288,  0.2345],
        [ 0.2303, -1.1229, -0.1863]])
After uniform_(0, 1):
tensor([[0.8694, 0.5677, 0.7411],
        [0.4294, 0.8854, 0.5739]])
```

#### `x.add_(y), x.mul_(y)  # in-place ops`

*In-place arithmetic. Modifies tensor directly, saves memory.*

**Example:**

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Original: {x}")
x.add_(10)
print(f"After add_(10): {x}")
x.mul_(2)
print(f"After mul_(2): {x}")
```

**Output:**

```
Original: tensor([1., 2., 3.])
After add_(10): tensor([11., 12., 13.])
After mul_(2): tensor([22., 24., 26.])
```

---
