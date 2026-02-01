# Easy Mathematics v1.0

**Mathematical Symbols & Formulas Reference**

*by jw*

This book covers essential mathematical concepts with PyTorch implementations:

- **Greek Letters** - Commonly used symbols
- **Formulas** - Key equations and identities
- **Code Examples** - PyTorch implementations

*Using PyTorch 2.8.0*

---

## Table of Contents

0. [Mathematical Constants](#mathematical-constants)
1. [Greek Letters (Lowercase)](#greek-letters-lowercase)
2. [Greek Letters (Uppercase)](#greek-letters-uppercase)
3. [Mathematical Operators](#mathematical-operators)
4. [Calculus Fundamentals](#calculus-fundamentals)
5. [Differential Equations](#differential-equations)
6. [Linear Algebra](#linear-algebra)
7. [Vector Calculus](#vector-calculus)
8. [Complex Analysis](#complex-analysis)
9. [Statistics & Probability](#statistics--probability)
10. [Series & Sequences](#series--sequences)
11. [Trigonometry](#trigonometry)
12. [Number Theory](#number-theory)
13. [Fourier Analysis](#fourier-analysis)
14. [Special Functions](#special-functions)
15. [Optimization](#optimization)
16. [Physics Formulas](#physics-formulas)

---

## Mathematical Constants

Important mathematical constants used throughout mathematics and physics.

```python
import torch
import math

print("=== Mathematical Constants ===")
print()
print(f"pi (pi)        = {math.pi:.15f}")
print(f"e (Euler)      = {math.e:.15f}")
print(f"phi (golden)   = {(1 + math.sqrt(5)) / 2:.15f}")
print(f"sqrt(2)        = {math.sqrt(2):.15f}")
print(f"sqrt(3)        = {math.sqrt(3):.15f}")
print(f"ln(2)          = {math.log(2):.15f}")
print(f"Euler-Mascheroni (gamma) ~ 0.5772156649...")
print()

# Verify some relationships
print("=== Relationships ===")
print(f"e^(i*pi) + 1 = {(math.e ** (1j * math.pi) + 1).real:.2e} (Euler's Identity)")
print(f"phi^2 - phi - 1 = {((1+math.sqrt(5))/2)**2 - (1+math.sqrt(5))/2 - 1:.2e}")
print(f"phi = 1 + 1/phi: {(1 + 2/(1+math.sqrt(5))):.15f}")
```

**Output:**

```
=== Mathematical Constants ===

pi (pi)        = 3.141592653589793
e (Euler)      = 2.718281828459045
phi (golden)   = 1.618033988749895
sqrt(2)        = 1.414213562373095
sqrt(3)        = 1.732050807568877
ln(2)          = 0.693147180559945
Euler-Mascheroni (gamma) ~ 0.5772156649...

=== Relationships ===
e^(i*pi) + 1 = 0.00e+00 (Euler's Identity)
phi^2 - phi - 1 = 0.00e+00
phi = 1 + 1/phi: 1.618033988749895
```

---

## α Greek Letters (Lowercase)

- alpha (alpha) - angles, coefficients
- beta (beta) - angles, parameters
- gamma (gamma) - Lorentz factor
- delta (delta) - small change, Dirac delta
- epsilon (epsilon) - small quantity, permittivity
- zeta (zeta) - Riemann zeta function
- eta (eta) - efficiency, viscosity
- theta (theta) - angles, temperature
- iota (iota) - imaginary unit (sometimes)
- kappa (kappa) - curvature, conductivity
- lambda (lambda) - wavelength, eigenvalue
- mu (mu) - mean, micro-, permeability
- nu (nu) - frequency, degrees of freedom
- xi (xi) - random variable
- pi (pi) - 3.14159... circle constant
- rho (rho) - density, correlation
- sigma (sigma) - std deviation, cross-section
- tau (tau) - time constant, torque
- phi (phi) - golden ratio, potential
- chi (chi) - chi-squared distribution
- psi (psi) - wave function
- omega (omega) - angular frequency

---

## Ω Greek Letters (Uppercase)

- Gamma - gamma function, Christoffel symbols
- Delta - change, Laplacian operator
- Theta - Big-O notation variant
- Lambda - cosmological constant
- Xi - grand canonical ensemble
- Pi - product notation
- Sigma - summation notation
- Phi - normal distribution CDF
- Psi - wave function (quantum)
- Omega - ohm, solid angle

---

## ∑ Mathematical Operators

- Summation: sum_i a_i = a_1 + a_2 + ... + a_n
- Product: prod_i a_i = a_1 * a_2 * ... * a_n
- Integral: integral f(x)dx
- Contour integral: closed integral f(z)dz
- Partial derivative: df/dx
- Nabla/Gradient: grad f = (df/dx, df/dy, df/dz)
- Laplacian: laplacian f = d2f/dx2 + d2f/dy2 + d2f/dz2
- Curl: curl F = rot F
- Divergence: div F
- Tensor product: A tensor B
- Direct sum: V direct_sum W
- Element of: x in A
- For all: forall x in R
- Exists: exists x : P(x)
- Empty set
- Infinity
- Approximately equal
- Congruent/Identical
- Proportional to
- Perpendicular
- Parallel

---

## ∫ Calculus Fundamentals

- Derivative: f'(x) = lim_{h->0} [f(x+h) - f(x)]/h
- Chain Rule: (f o g)'(x) = f'(g(x)) * g'(x)
- Product Rule: (fg)' = f'g + fg'
- Quotient Rule: (f/g)' = (f'g - fg')/g^2
- Integration by Parts: integral u dv = uv - integral v du
- Fundamental Theorem: integral_a^b f'(x)dx = f(b) - f(a)
- Taylor Series: f(x) = sum f^(n)(a)/n! * (x-a)^n
- Power Rule: d/dx(x^n) = nx^(n-1)
- Exponential: d/dx(e^x) = e^x
- Logarithm: d/dx(ln x) = 1/x
- Trigonometric: d/dx(sin x) = cos x
- Arc Length: L = integral sqrt(1 + [f'(x)]^2) dx
- Volume of Revolution: V = pi * integral [f(x)]^2 dx

### Code Examples

#### Derivative Numerical

*PyTorch autograd computes derivatives automatically using the chain rule.*

```python
import torch

# Numerical derivative using PyTorch autograd
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x^3

# Compute derivative: dy/dx = 3x^2
y.backward()
print(f"x = {x.item()}")
print(f"y = x^3 = {y.item()}")
print(f"dy/dx (autograd) = {x.grad.item()}")
print(f"dy/dx (analytical: 3x^2) = {3 * 2**2}")
```

**Output:**

```
x = 2.0
y = x^3 = 8.0
dy/dx (autograd) = 12.0
dy/dx (analytical: 3x^2) = 12
```

#### Chain Rule

*The chain rule is automatically applied by autograd for composite functions.*

```python
import torch

# Chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
# Example: f(x) = sin(x^2), so g(x) = x^2, f(u) = sin(u)
x = torch.tensor(1.0, requires_grad=True)
y = torch.sin(x ** 2)

y.backward()
print(f"x = {x.item()}")
print(f"y = sin(x^2) = {y.item():.6f}")
print(f"dy/dx (autograd) = {x.grad.item():.6f}")
# Analytical: cos(x^2) * 2x
analytical = torch.cos(x.detach() ** 2) * 2 * x.detach()
print(f"dy/dx (analytical: 2x*cos(x^2)) = {analytical.item():.6f}")
```

**Output:**

```
x = 1.0
y = sin(x^2) = 0.841471
dy/dx (autograd) = 1.080605
dy/dx (analytical: 2x*cos(x^2)) = 1.080605
```

#### Taylor Series

*Taylor series provides polynomial approximations of functions.*

```python
import torch
import math

# Taylor series approximation of e^x around x=0
# e^x = 1 + x + x^2/2! + x^3/3! + ...
x = torch.tensor(1.0)
n_terms = 10

approx = torch.tensor(0.0)
for n in range(n_terms):
    approx += (x ** n) / math.factorial(n)

print(f"x = {x.item()}")
print(f"Taylor series (10 terms): {approx.item():.10f}")
print(f"torch.exp(x): {torch.exp(x).item():.10f}")
print(f"math.e: {math.e:.10f}")
```

**Output:**

```
x = 1.0
Taylor series (10 terms): 2.7182817459
torch.exp(x): 2.7182817459
math.e: 2.7182818285
```

---

## ∂ Differential Equations

- 1st Order Linear: dy/dx + P(x)y = Q(x)
- Integrating Factor: mu(x) = e^{integral P(x)dx}
- Separable: dy/dx = g(x)h(y)
- Exact: M(x,y)dx + N(x,y)dy = 0
- 2nd Order Linear: y'' + p(x)y' + q(x)y = r(x)
- Characteristic Eq: ar^2 + br + c = 0
- Homogeneous Solution: y_h = c1*e^(r1*x) + c2*e^(r2*x)
- Wave Equation: d2u/dt2 = c^2 * laplacian u
- Heat Equation: du/dt = alpha * laplacian u
- Laplace's Equation: laplacian u = 0
- Schrodinger: i*hbar * dpsi/dt = H*psi
- Euler-Lagrange: dL/dq - d/dt(dL/dq_dot) = 0

---

## ⊗ Linear Algebra

- Matrix Product: (AB)_ij = sum_k A_ik * B_kj
- Determinant 2x2: |A| = ad - bc
- Determinant 3x3: Sarrus' rule
- Inverse: A^(-1) * A = A * A^(-1) = I
- Transpose: (A^T)_ij = A_ji
- Eigenvalue: Av = lambda * v
- Characteristic Poly: det(A - lambda*I) = 0
- Trace: tr(A) = sum_i A_ii = sum lambda_i
- Rank: dim(Im(A))
- Null Space: ker(A) = {x : Ax = 0}
- Gram-Schmidt: Orthogonalization
- SVD: A = U * Sigma * V^T
- QR Decomposition: A = QR

### Code Examples

#### Matrix Operations

*Matrix multiplication follows the rule (AB)_ij = sum_k A_ik * B_kj.*

```python
import torch

# Matrix operations
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Matrix multiplication: (AB)_ij = sum_k A_ik * B_kj
print("\nA @ B (matrix multiplication):")
print(A @ B)

# Element-wise multiplication (Hadamard product)
print("\nA * B (element-wise):")
print(A * B)
```

**Output:**

```
Matrix A:
tensor([[1., 2.],
        [3., 4.]])

Matrix B:
tensor([[5., 6.],
        [7., 8.]])

A @ B (matrix multiplication):
tensor([[19., 22.],
        [43., 50.]])

A * B (element-wise):
tensor([[ 5., 12.],
        [21., 32.]])
```

#### Determinant Inverse

*For 2x2 matrix [[a,b],[c,d]], det = ad-bc. A^(-1) exists iff det != 0.*

```python
import torch

# Determinant and inverse
A = torch.tensor([[1., 2.], [3., 4.]])

print("Matrix A:")
print(A)

# Determinant: |A| = ad - bc
det = torch.det(A)
print(f"\nDeterminant: {det.item():.4f}")
print(f"Manual: 1*4 - 2*3 = {1*4 - 2*3}")

# Inverse: A^(-1) * A = I
A_inv = torch.inverse(A)
print("\nInverse A^(-1):")
print(A_inv)

print("\nVerify A @ A^(-1) = I:")
print(A @ A_inv)
```

**Output:**

```
Matrix A:
tensor([[1., 2.],
        [3., 4.]])

Determinant: -2.0000
Manual: 1*4 - 2*3 = -2

Inverse A^(-1):
tensor([[-2.0000,  1.0000],
        [ 1.5000, -0.5000]])

Verify A @ A^(-1) = I:
tensor([[1., 0.],
        [0., 1.]])
```

#### Eigenvalues

*Eigenvalues satisfy det(A - lambda*I) = 0. Eigenvectors satisfy Av = lambda*v.*

```python
import torch

# Eigenvalues and eigenvectors: Av = lambda * v
A = torch.tensor([[4., 2.], [1., 3.]])

print("Matrix A:")
print(A)

# Compute eigenvalues
eigenvalues = torch.linalg.eigvalsh(A.float())  # For symmetric-like
print(f"\nEigenvalues: {eigenvalues}")

# Full decomposition
eigvals, eigvecs = torch.linalg.eig(A)
print(f"\nEigenvalues (complex): {eigvals}")
print("Eigenvectors:")
print(eigvecs)
```

**Output:**

```
Matrix A:
tensor([[4., 2.],
        [1., 3.]])

Eigenvalues: tensor([2.3820, 4.6180])

Eigenvalues (complex): tensor([5.+0.j, 2.+0.j])
Eigenvectors:
tensor([[ 0.8944+0.j, -0.7071+0.j],
        [ 0.4472+0.j,  0.7071+0.j]])
```

#### Svd Decomposition

*SVD decomposes any matrix into orthogonal matrices U, V and singular values S.*

```python
import torch

# SVD: A = U * Sigma * V^T
A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

print("Matrix A:")
print(A)
print(f"Shape: {A.shape}")

U, S, Vh = torch.linalg.svd(A)
print(f"\nU shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"Vh shape: {Vh.shape}")

# Reconstruct
reconstructed = U @ torch.diag(S) @ Vh[:S.shape[0], :]
print("\nReconstructed A:")
print(reconstructed)
```

**Output:**

```
Matrix A:
tensor([[1., 2., 3.],
        [4., 5., 6.]])
Shape: torch.Size([2, 3])

U shape: torch.Size([2, 2])
S (singular values): tensor([9.5080, 0.7729])
Vh shape: torch.Size([3, 3])

Reconstructed A:
tensor([[1.0000, 2.0000, 3.0000],
        [4.0000, 5.0000, 6.0000]])
```

---

## ∇ Vector Calculus

- Gradient: grad f = (df/dx)i + (df/dy)j + (df/dz)k
- Divergence: div F = dFx/dx + dFy/dy + dFz/dz
- Curl: curl F = determinant form
- Line Integral: integral_C F . dr
- Surface Integral: double_integral_S F . n dS
- Green's Theorem: closed_integral_C F.dr = double_integral_D (dQ/dx - dP/dy)dA
- Stokes' Theorem: closed_integral_C F.dr = double_integral_S (curl F).n dS
- Divergence Theorem: double_integral_S F.n dS = triple_integral_V div F dV
- Directional Derivative: D_u f = grad f . u

---

## ℂ Complex Analysis

- Euler's Formula: e^(i*theta) = cos(theta) + i*sin(theta)
- Complex Conjugate: z_bar = a - bi
- Modulus: |z| = sqrt(a^2 + b^2)
- Argument: arg(z) = arctan(b/a)
- De Moivre: (cos theta + i sin theta)^n = cos(n*theta) + i*sin(n*theta)
- Cauchy-Riemann: du/dx = dv/dy, du/dy = -dv/dx
- Cauchy Integral: f(z0) = 1/(2*pi*i) * closed_integral_C f(z)/(z-z0) dz
- Residue Theorem: closed_integral_C f(z)dz = 2*pi*i * sum Res(f, z_k)
- Laurent Series: f(z) = sum a_n * (z-z0)^n
- Analytic Continuation
- Riemann Surfaces

### Code Examples

#### Euler Formula

*Euler's formula connects exponentials with trigonometry in the complex plane.*

```python
import torch
import math

# Euler's Formula: e^(i*theta) = cos(theta) + i*sin(theta)
theta = torch.tensor(math.pi / 4)  # 45 degrees

# Using torch complex exponential
z = torch.exp(1j * theta)
print(f"theta = pi/4 = {theta.item():.6f}")
print(f"\ne^(i*theta) = {z}")
print(f"Real part: {z.real.item():.6f}")
print(f"Imag part: {z.imag.item():.6f}")

# Verify with cos and sin
print(f"\ncos(theta) = {math.cos(theta.item()):.6f}")
print(f"sin(theta) = {math.sin(theta.item()):.6f}")

# Famous identity: e^(i*pi) + 1 = 0
euler_identity = torch.exp(1j * torch.tensor(math.pi)) + 1
print(f"\nEuler's Identity: e^(i*pi) + 1 = {euler_identity}")
print(f"(Should be ~0)")
```

**Output:**

```
theta = pi/4 = 0.785398

e^(i*theta) = (0.7071067690849304+0.7071067690849304j)
Real part: 0.707107
Imag part: 0.707107

cos(theta) = 0.707107
sin(theta) = 0.707107

Euler's Identity: e^(i*pi) + 1 = -8.742277657347586e-08j
(Should be ~0)
```

---

## μ Statistics & Probability

- Mean: mu = (1/n) * sum x_i
- Variance: sigma^2 = E[(X - mu)^2]
- Standard Deviation: sigma = sqrt(sigma^2)
- Covariance: Cov(X,Y) = E[(X-mu_x)(Y-mu_y)]
- Correlation: rho = Cov(X,Y)/(sigma_x * sigma_y)
- Bayes' Theorem: P(A|B) = P(B|A)*P(A)/P(B)
- Normal Distribution: f(x) = (1/(sigma*sqrt(2*pi))) * e^(-(x-mu)^2/(2*sigma^2))
- Binomial: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
- Poisson: P(X=k) = (lambda^k * e^(-lambda))/k!
- Central Limit Theorem: X_bar ~ N(mu, sigma^2/n)
- Chi-squared: chi^2 = sum (O_i - E_i)^2/E_i
- t-distribution: t = (X_bar - mu)/(s/sqrt(n))

### Code Examples

#### Mean Variance

*Variance measures the spread of data around the mean.*

```python
import torch

# Mean and Variance
data = torch.tensor([2., 4., 4., 4., 5., 5., 7., 9.])

# Mean: mu = (1/n) * sum(x_i)
mean = data.mean()
print(f"Data: {data.tolist()}")
print(f"Mean (mu): {mean.item():.4f}")

# Variance: sigma^2 = E[(X - mu)^2]
variance = data.var(unbiased=False)  # Population variance
print(f"Variance (sigma^2): {variance.item():.4f}")

# Standard deviation: sigma = sqrt(variance)
std = data.std(unbiased=False)
print(f"Std deviation (sigma): {std.item():.4f}")

# Manual calculation
manual_var = ((data - mean) ** 2).mean()
print(f"\nManual variance: {manual_var.item():.4f}")
```

**Output:**

```
Data: [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
Mean (mu): 5.0000
Variance (sigma^2): 4.0000
Std deviation (sigma): 2.0000

Manual variance: 4.0000
```

#### Covariance Correlation

*Correlation normalizes covariance to [-1, 1] range.*

```python
import torch

# Covariance and Correlation
X = torch.tensor([1., 2., 3., 4., 5.])
Y = torch.tensor([2., 4., 5., 4., 5.])

# Covariance: Cov(X,Y) = E[(X-mu_x)(Y-mu_y)]
mean_x, mean_y = X.mean(), Y.mean()
cov = ((X - mean_x) * (Y - mean_y)).mean()

print(f"X: {X.tolist()}")
print(f"Y: {Y.tolist()}")
print(f"Covariance: {cov.item():.4f}")

# Correlation: rho = Cov(X,Y) / (sigma_x * sigma_y)
std_x = X.std(unbiased=False)
std_y = Y.std(unbiased=False)
correlation = cov / (std_x * std_y)
print(f"Correlation: {correlation.item():.4f}")
```

**Output:**

```
X: [1.0, 2.0, 3.0, 4.0, 5.0]
Y: [2.0, 4.0, 5.0, 4.0, 5.0]
Covariance: 1.2000
Correlation: 0.7746
```

#### Normal Distribution

*The normal distribution is fundamental in statistics and machine learning.*

```python
import torch
import math

# Normal (Gaussian) distribution
# f(x) = (1/(sigma*sqrt(2*pi))) * exp(-(x-mu)^2/(2*sigma^2))
mu = 0.0
sigma = 1.0
x = torch.linspace(-3, 3, 7)

def normal_pdf(x, mu, sigma):
    coef = 1 / (sigma * math.sqrt(2 * math.pi))
    exp_term = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coef * exp_term

pdf = normal_pdf(x, mu, sigma)
print(f"x values: {x.tolist()}")
print(f"PDF values:")
for xi, pi in zip(x.tolist(), pdf.tolist()):
    print(f"  f({xi:5.2f}) = {pi:.6f}")
```

**Output:**

```
x values: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
PDF values:
  f(-3.00) = 0.004432
  f(-2.00) = 0.053991
  f(-1.00) = 0.241971
  f( 0.00) = 0.398942
  f( 1.00) = 0.241971
  f( 2.00) = 0.053991
  f( 3.00) = 0.004432
```

#### Bayes Theorem

*Bayes' theorem updates probabilities given new evidence.*

```python
import torch

# Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
# Example: Medical test
# P(Disease) = 0.01 (1% have disease)
# P(Positive|Disease) = 0.99 (99% true positive rate)
# P(Positive|No Disease) = 0.05 (5% false positive rate)

P_disease = torch.tensor(0.01)
P_positive_given_disease = torch.tensor(0.99)
P_positive_given_no_disease = torch.tensor(0.05)

# P(Positive) = P(+|D)*P(D) + P(+|~D)*P(~D)
P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_no_disease * (1 - P_disease))

# P(Disease|Positive) = P(+|D) * P(D) / P(+)
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print("Bayes' Theorem Example: Medical Test")
print(f"Prior P(Disease): {P_disease.item():.2%}")
print(f"P(Positive|Disease): {P_positive_given_disease.item():.2%}")
print(f"P(Positive|No Disease): {P_positive_given_no_disease.item():.2%}")
print(f"\nP(Positive): {P_positive.item():.4f}")
print(f"P(Disease|Positive): {P_disease_given_positive.item():.2%}")
print("\nEven with positive test, only ~17% chance of disease!")
```

**Output:**

```
Bayes' Theorem Example: Medical Test
Prior P(Disease): 1.00%
P(Positive|Disease): 99.00%
P(Positive|No Disease): 5.00%

P(Positive): 0.0594
P(Disease|Positive): 16.67%

Even with positive test, only ~17% chance of disease!
```

---

## Σ Series & Sequences

- Arithmetic: a_n = a_1 + (n-1)d
- Geometric: a_n = a_1 * r^(n-1)
- Arithmetic Sum: S_n = n(a_1 + a_n)/2
- Geometric Sum: S_n = a_1(1 - r^n)/(1 - r)
- Infinite Geometric: S = a_1/(1 - r), |r| < 1
- Harmonic Series: sum(1/n) diverges
- p-Series: sum(1/n^p) converges if p > 1
- Alternating Series: sum(-1)^n * a_n
- Ratio Test: lim|a_(n+1)/a_n| < 1 -> converges
- Root Test: lim|a_n|^(1/n) < 1 -> converges
- Integral Test: integral f(x)dx converges <-> sum f(n) converges
- Comparison Test

---

## ∿ Trigonometry

- Pythagorean: sin^2(theta) + cos^2(theta) = 1
- tan^2(theta) + 1 = sec^2(theta)
- 1 + cot^2(theta) = csc^2(theta)
- Double Angle: sin(2*theta) = 2*sin(theta)*cos(theta)
- cos(2*theta) = cos^2(theta) - sin^2(theta) = 2*cos^2(theta) - 1
- Half Angle: sin^2(theta/2) = (1 - cos(theta))/2
- Sum/Difference: sin(a +/- b) = sin(a)cos(b) +/- cos(a)sin(b)
- cos(a +/- b) = cos(a)cos(b) -/+ sin(a)sin(b)
- Product to Sum: sin(A)*sin(B) = (1/2)[cos(A-B) - cos(A+B)]
- Law of Sines: a/sin(A) = b/sin(B) = c/sin(C)
- Law of Cosines: c^2 = a^2 + b^2 - 2ab*cos(C)
- Inverse: sin^(-1)(x) + cos^(-1)(x) = pi/2

### Code Examples

#### Trig Identities

*Trigonometric identities are fundamental for simplifying expressions.*

```python
import torch
import math

theta = torch.tensor(math.pi / 6)  # 30 degrees

print(f"theta = pi/6 (30 degrees)")
print()

# Pythagorean identity: sin^2 + cos^2 = 1
sin_t = torch.sin(theta)
cos_t = torch.cos(theta)
identity = sin_t**2 + cos_t**2
print(f"sin^2(theta) + cos^2(theta) = {identity.item():.10f}")

# Double angle formulas
sin_2t = torch.sin(2 * theta)
double_angle = 2 * sin_t * cos_t
print(f"\nsin(2*theta) = {sin_2t.item():.6f}")
print(f"2*sin(theta)*cos(theta) = {double_angle.item():.6f}")

# Verify cos(2*theta) = cos^2 - sin^2
cos_2t = torch.cos(2 * theta)
cos_double = cos_t**2 - sin_t**2
print(f"\ncos(2*theta) = {cos_2t.item():.6f}")
print(f"cos^2(theta) - sin^2(theta) = {cos_double.item():.6f}")
```

**Output:**

```
theta = pi/6 (30 degrees)

sin^2(theta) + cos^2(theta) = 1.0000000000

sin(2*theta) = 0.866025
2*sin(theta)*cos(theta) = 0.866025

cos(2*theta) = 0.500000
cos^2(theta) - sin^2(theta) = 0.500000
```

---

## ∮ Number Theory

- Prime Number Theorem: pi(x) ~ x/ln(x)
- Euler's Totient: phi(n) = n * prod(1 - 1/p)
- Fermat's Little: a^(p-1) = 1 (mod p)
- Euler's Theorem: a^(phi(n)) = 1 (mod n)
- Chinese Remainder Theorem
- Quadratic Reciprocity: (p/q)(q/p) = (-1)^((p-1)(q-1)/4)
- Divisor Function: sigma(n) = sum_{d|n} d
- Mobius Function: mu(n)
- Riemann Zeta: zeta(s) = sum(1/n^s)
- Goldbach Conjecture
- Twin Prime Conjecture

---

## ∫ Fourier Analysis

- Fourier Series: f(x) = a0/2 + sum[a_n*cos(nx) + b_n*sin(nx)]
- Coefficients: a_n = (2/pi) * integral f(x)cos(nx)dx
- b_n = (2/pi) * integral f(x)sin(nx)dx
- Complex Form: f(x) = sum c_n * e^(inx)
- Fourier Transform: F(omega) = integral f(t)*e^(-i*omega*t)dt
- Inverse Transform: f(t) = (1/2*pi) * integral F(omega)*e^(i*omega*t)d_omega
- Convolution: (f * g)(t) = integral f(tau)*g(t-tau)d_tau
- Parseval's Identity: integral|f(x)|^2 dx = (1/2*pi) * integral|F(omega)|^2 d_omega
- DFT: X_k = sum x_n * e^(-2*pi*i*k*n/N)
- FFT: O(N log N) algorithm

### Code Examples

#### Fourier Transform

*FFT decomposes signals into frequency components in O(N log N) time.*

```python
import torch
import math

# Discrete Fourier Transform (DFT)
# X_k = sum_n x_n * e^(-2*pi*i*k*n/N)

# Create a simple signal: sum of two sinusoids
N = 64
t = torch.linspace(0, 1, N)
freq1, freq2 = 5, 12  # Hz
signal = torch.sin(2 * math.pi * freq1 * t) + 0.5 * torch.sin(2 * math.pi * freq2 * t)

# Compute FFT
fft_result = torch.fft.fft(signal)
magnitudes = torch.abs(fft_result)
freqs = torch.fft.fftfreq(N, 1/N)

# Find peaks (top frequencies)
top_idx = torch.argsort(magnitudes[:N//2], descending=True)[:3]
print("Signal: sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)")
print(f"\nTop frequency components:")
for idx in top_idx:
    print(f"  Freq: {freqs[idx].item():.1f} Hz, Magnitude: {magnitudes[idx].item():.2f}")
```

**Output:**

```
Signal: sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)

Top frequency components:
  Freq: 5.0 Hz, Magnitude: 31.63
  Freq: 12.0 Hz, Magnitude: 14.37
  Freq: 13.0 Hz, Magnitude: 4.18
```

---

## Γ Special Functions

- Gamma: Gamma(n) = (n-1)! = integral_0^inf t^(n-1)*e^(-t)dt
- Beta: B(x,y) = integral_0^1 t^(x-1)*(1-t)^(y-1)dt
- Error Function: erf(x) = (2/sqrt(pi)) * integral_0^x e^(-t^2)dt
- Bessel Functions: J_n(x)
- Legendre Polynomials: P_n(x)
- Hermite Polynomials: H_n(x)
- Laguerre Polynomials: L_n(x)
- Chebyshev Polynomials: T_n(x)
- Hypergeometric: 2F1(a,b;c;z)
- Elliptic Integrals: K(k), E(k)
- Riemann Zeta: zeta(s) = sum(1/n^s)

### Code Examples

#### Gamma Function

*The gamma function extends factorials to non-integers.*

```python
import torch
import math

# Gamma function: Gamma(n) = (n-1)! for positive integers
# Gamma(n) = integral_0^inf t^(n-1) * e^(-t) dt

# For integers, Gamma(n) = (n-1)!
print("Gamma Function: Gamma(n) = (n-1)!")
print()
for n in range(1, 7):
    gamma_n = math.gamma(n)
    factorial = math.factorial(n-1)
    print(f"Gamma({n}) = {gamma_n:.4f} = {n-1}! = {factorial}")

# Gamma(1/2) = sqrt(pi)
gamma_half = math.gamma(0.5)
print(f"\nGamma(1/2) = {gamma_half:.6f}")
print(f"sqrt(pi) = {math.sqrt(math.pi):.6f}")
```

**Output:**

```
Gamma Function: Gamma(n) = (n-1)!

Gamma(1) = 1.0000 = 0! = 1
Gamma(2) = 1.0000 = 1! = 1
Gamma(3) = 2.0000 = 2! = 2
Gamma(4) = 6.0000 = 3! = 6
Gamma(5) = 24.0000 = 4! = 24
Gamma(6) = 120.0000 = 5! = 120

Gamma(1/2) = 1.772454
sqrt(pi) = 1.772454
```

---

## ⊕ Optimization

- Gradient Descent: x_(n+1) = x_n - alpha * grad f(x_n)
- Newton's Method: x_(n+1) = x_n - f(x_n)/f'(x_n)
- Lagrange Multipliers: grad f = lambda * grad g
- KKT Conditions: Stationarity, Primal/Dual feasibility
- Convex Optimization: f(theta*x + (1-theta)*y) <= theta*f(x) + (1-theta)*f(y)
- Linear Programming: max c^T*x s.t. Ax <= b
- Quadratic Programming: min (1/2)*x^T*Q*x + c^T*x
- Conjugate Gradient Method
- BFGS Algorithm
- Interior Point Methods
- Simplex Algorithm

### Code Examples

#### Gradient Descent

*Gradient descent iteratively moves toward the minimum by following negative gradient.*

```python
import torch

# Gradient Descent: x_(n+1) = x_n - alpha * grad f(x_n)
# Minimize f(x) = x^2 - 4x + 4 = (x-2)^2
# Minimum at x = 2

x = torch.tensor(0.0, requires_grad=True)
lr = 0.1  # learning rate (alpha)

print("Gradient Descent to minimize f(x) = (x-2)^2")
print(f"Starting at x = {x.item():.4f}")
print()

for i in range(10):
    # Forward: compute f(x)
    y = (x - 2) ** 2

    # Backward: compute gradient
    y.backward()

    # Update: x = x - lr * grad
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()

    if i % 2 == 0:
        print(f"Step {i}: x = {x.item():.4f}, f(x) = {((x-2)**2).item():.6f}")

print(f"\nFinal x = {x.item():.4f} (optimal = 2.0)")
```

**Output:**

```
Gradient Descent to minimize f(x) = (x-2)^2
Starting at x = 0.0000

Step 0: x = 0.4000, f(x) = 2.560000
Step 2: x = 0.9760, f(x) = 1.048576
Step 4: x = 1.3446, f(x) = 0.429497
Step 6: x = 1.5806, f(x) = 0.175922
Step 8: x = 1.7316, f(x) = 0.072058

Final x = 1.7853 (optimal = 2.0)
```

#### Newtons Method

*Newton's method converges quadratically for well-behaved functions.*

```python
import torch

# Newton's Method: x_(n+1) = x_n - f(x_n) / f'(x_n)
# Find root of f(x) = x^2 - 2 (i.e., sqrt(2))

x = torch.tensor(1.0, requires_grad=True)
print("Newton's Method to find sqrt(2)")
print(f"Solving x^2 - 2 = 0")
print(f"Starting at x = {x.item():.4f}")
print()

for i in range(6):
    # f(x) = x^2 - 2
    y = x ** 2 - 2

    # f'(x) = 2x
    y.backward()
    f_prime = x.grad.item()

    # Newton update: x = x - f(x)/f'(x)
    with torch.no_grad():
        x -= y / f_prime
        x.grad.zero_()

    print(f"Step {i}: x = {x.item():.10f}")

print(f"\nResult: {x.item():.10f}")
print(f"Actual sqrt(2): {2**0.5:.10f}")
```

**Output:**

```
Newton's Method to find sqrt(2)
Solving x^2 - 2 = 0
Starting at x = 1.0000

Step 0: x = 1.5000000000
Step 1: x = 1.4166666269
Step 2: x = 1.4142156839
Step 3: x = 1.4142135382
Step 4: x = 1.4142135382
Step 5: x = 1.4142135382

Result: 1.4142135382
Actual sqrt(2): 1.4142135624
```

---

## ⚛ Physics Formulas

- Newton's 2nd Law: F = ma = dp/dt
- Energy: E = mc^2
- Momentum: p = mv
- Angular Momentum: L = r x p
- Schrodinger Eq: i*hbar*dpsi/dt = H*psi
- Maxwell's Equations:
-   div E = rho/epsilon_0
-   div B = 0
-   curl E = -dB/dt
-   curl B = mu_0*(J + epsilon_0*dE/dt)
- Lorentz Transform: x' = gamma*(x - vt)
- Planck's Law: E = h*nu
- Heisenberg: Delta_x * Delta_p >= hbar/2

### Code Examples

#### Wave Equation

*The wave equation describes propagation of waves through a medium.*

```python
import torch

# Wave equation: d2u/dt2 = c^2 * d2u/dx2
# Simple numerical solution using finite differences

# Parameters
nx, nt = 50, 100
dx, dt = 0.1, 0.01
c = 1.0  # wave speed
r = (c * dt / dx) ** 2

# Initial condition: Gaussian pulse
x = torch.linspace(0, (nx-1)*dx, nx)
u = torch.exp(-((x - 2.5) ** 2))

# Store solutions
u_prev = u.clone()
u_curr = u.clone()

print("Wave Equation Simulation")
print(f"Wave speed c = {c}")
print(f"Grid: {nx} points, {nt} time steps")
print(f"\nInitial pulse at x=2.5")
print(f"Max amplitude at t=0: {u_curr.max().item():.4f}")

# Time stepping
for t in range(nt):
    u_next = torch.zeros_like(u_curr)
    for i in range(1, nx-1):
        u_next[i] = 2*u_curr[i] - u_prev[i] + r*(u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
    u_prev = u_curr.clone()
    u_curr = u_next.clone()

    if t == nt//2:
        print(f"Max amplitude at t={t}: {u_curr.max().item():.4f}")

print(f"Max amplitude at t={nt}: {u_curr.max().item():.4f}")
print("\nWave propagates and spreads over time.")
```

**Output:**

```
Wave Equation Simulation
Wave speed c = 1.0
Grid: 50 points, 100 time steps

Initial pulse at x=2.5
Max amplitude at t=0: 1.0000
Max amplitude at t=50: 0.7679
Max amplitude at t=100: 0.5088

Wave propagates and spreads over time.
```

---
