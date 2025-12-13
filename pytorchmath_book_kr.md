# 쉬운 수학 v1.0

**수학 기호 & 공식 참조서**

*by jw*

이 책은 PyTorch 구현과 함께 필수 수학 개념을 다룹니다:

- **그리스 문자** - 자주 사용되는 기호
- **공식** - 핵심 방정식과 항등식
- **코드 예제** - PyTorch 구현

*PyTorch 2.8.0 사용*

---

## 목차

0. [수학 상수](#수학-상수)
1. [그리스 문자 (소문자)](#그리스-문자-소문자)
2. [그리스 문자 (대문자)](#그리스-문자-대문자)
3. [수학 연산자](#수학-연산자)
4. [미적분 기초](#미적분-기초)
5. [미분방정식](#미분방정식)
6. [선형대수](#선형대수)
7. [벡터 해석](#벡터-해석)
8. [복소해석](#복소해석)
9. [통계 & 확률](#통계--확률)
10. [급수 & 수열](#급수--수열)
11. [삼각함수](#삼각함수)
12. [정수론](#정수론)
13. [푸리에 해석](#푸리에-해석)
14. [특수 함수](#특수-함수)
15. [최적화](#최적화)
16. [물리 공식](#물리-공식)

---

## 수학 상수

수학과 물리학에서 사용되는 중요한 수학 상수들입니다.

```python
import torch
import math

print("=== 수학 상수 ===")
print()
print(f"π (파이)        = {math.pi:.15f}")
print(f"e (오일러 수)   = {math.e:.15f}")
print(f"φ (황금비)      = {(1 + math.sqrt(5)) / 2:.15f}")
print(f"√2              = {math.sqrt(2):.15f}")
print(f"√3              = {math.sqrt(3):.15f}")
print(f"ln(2)           = {math.log(2):.15f}")
print(f"오일러-마스케로니 상수 (γ) ≈ 0.5772156649...")
print()

# 관계 검증
print("=== 관계식 ===")
print(f"e^(i*π) + 1 = {(math.e ** (1j * math.pi) + 1).real:.2e} (오일러 항등식)")
print(f"φ^2 - φ - 1 = {((1+math.sqrt(5))/2)**2 - (1+math.sqrt(5))/2 - 1:.2e}")
print(f"φ = 1 + 1/φ: {(1 + 2/(1+math.sqrt(5))):.15f}")
```

**출력:**

```
=== 수학 상수 ===

π (파이)        = 3.141592653589793
e (오일러 수)   = 2.718281828459045
φ (황금비)      = 1.618033988749895
√2              = 1.414213562373095
√3              = 1.732050807568877
ln(2)           = 0.693147180559945
오일러-마스케로니 상수 (γ) ≈ 0.5772156649...

=== 관계식 ===
e^(i*π) + 1 = 0.00e+00 (오일러 항등식)
φ^2 - φ - 1 = 0.00e+00
φ = 1 + 1/φ: 1.618033988749895
```

---

## α 그리스 문자 (소문자)

- α (알파) - 각도, 계수
- β (베타) - 각도, 매개변수
- γ (감마) - 로렌츠 인자
- δ (델타) - 미소 변화, 디랙 델타
- ε (엡실론) - 미소량, 유전율
- ζ (제타) - 리만 제타 함수
- η (에타) - 효율, 점성
- θ (세타) - 각도, 온도
- ι (이오타) - 허수 단위 (가끔)
- κ (카파) - 곡률, 전도율
- λ (람다) - 파장, 고유값
- μ (뮤) - 평균, 마이크로, 투자율
- ν (뉴) - 주파수, 자유도
- ξ (크시) - 확률 변수
- π (파이) ≈ 3.14159... - 원주율
- ρ (로) - 밀도, 상관계수
- σ (시그마) - 표준편차, 단면적
- τ (타우) - 시간 상수, 토크
- φ (파이) - 황금비, 전위
- χ (카이) - 카이제곱 분포
- ψ (프사이) - 파동 함수
- ω (오메가) - 각진동수

---

## Ω 그리스 문자 (대문자)

- Γ (감마) - 감마 함수, 크리스토펠 기호
- Δ (델타) - 변화량, 라플라시안
- Θ (세타) - Big-O 표기법 변형
- Λ (람다) - 우주 상수
- Ξ (크시) - 대정준 앙상블
- Π (파이) - 곱 기호
- Σ (시그마) - 합 기호
- Φ (파이) - 정규분포 누적분포함수
- Ψ (프사이) - 파동 함수 (양자)
- Ω (오메가) - 옴, 입체각

---

## ∑ 수학 연산자

- ∑ (합): ∑ᵢ₌₁ⁿ aᵢ = a₁ + a₂ + ... + aₙ
- ∏ (곱): ∏ᵢ₌₁ⁿ aᵢ = a₁ × a₂ × ... × aₙ
- ∫ (적분): ∫ₐᵇ f(x)dx
- ∮ (선적분): ∮_C f(z)dz
- ∂ (편미분): ∂f/∂x
- ∇ (나블라/기울기): ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)
- ∇² (라플라시안): ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
- ∇× (회전): ∇×F = rot F
- ∇· (발산): ∇·F = div F
- ⊗ (텐서곱): A ⊗ B
- ⊕ (직합): V ⊕ W
- ∈ (원소): x ∈ A
- ∀ (모든): ∀x ∈ ℝ
- ∃ (존재): ∃x : P(x)
- ∅ (공집합)
- ∞ (무한대)
- ≈ (근사)
- ≡ (합동/항등)
- ∝ (비례)
- ⊥ (수직)
- ∥ (평행)

---

## ∫ 미적분 기초

- 도함수: f'(x) = lim_{h→0} [f(x+h) - f(x)]/h
- 연쇄 법칙: (f∘g)'(x) = f'(g(x)) · g'(x)
- 곱의 법칙: (fg)' = f'g + fg'
- 몫의 법칙: (f/g)' = (f'g - fg')/g²
- 부분 적분: ∫u dv = uv - ∫v du
- 미적분 기본정리: ∫ₐᵇ f'(x)dx = f(b) - f(a)
- 테일러 급수: f(x) = ∑_{n=0}^∞ f⁽ⁿ⁾(a)/n! · (x-a)ⁿ
- 거듭제곱 법칙: d/dx(xⁿ) = nxⁿ⁻¹
- 지수 함수: d/dx(eˣ) = eˣ
- 로그 함수: d/dx(ln x) = 1/x
- 삼각 함수: d/dx(sin x) = cos x
- 호의 길이: L = ∫ₐᵇ √(1 + [f'(x)]²) dx
- 회전체 부피: V = π∫ₐᵇ [f(x)]² dx

### 코드 예제

#### Derivative Numerical

*PyTorch autograd는 연쇄 법칙을 사용하여 자동으로 도함수를 계산합니다.*

```python
import torch

# PyTorch autograd를 사용한 수치 미분
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x^3

# 도함수 계산: dy/dx = 3x^2
y.backward()
print(f"x = {x.item()}")
print(f"y = x^3 = {y.item()}")
print(f"dy/dx (autograd) = {x.grad.item()}")
print(f"dy/dx (해석적: 3x^2) = {3 * 2**2}")
```

**출력:**

```
x = 2.0
y = x^3 = 8.0
dy/dx (autograd) = 12.0
dy/dx (해석적: 3x^2) = 12
```

#### Chain Rule

*연쇄 법칙은 합성 함수에 대해 autograd가 자동으로 적용합니다.*

```python
import torch

# 연쇄 법칙: d/dx[f(g(x))] = f'(g(x)) * g'(x)
# 예: f(x) = sin(x^2), g(x) = x^2, f(u) = sin(u)
x = torch.tensor(1.0, requires_grad=True)
y = torch.sin(x ** 2)

y.backward()
print(f"x = {x.item()}")
print(f"y = sin(x^2) = {y.item():.6f}")
print(f"dy/dx (autograd) = {x.grad.item():.6f}")
# 해석적: cos(x^2) * 2x
analytical = torch.cos(x.detach() ** 2) * 2 * x.detach()
print(f"dy/dx (해석적: 2x*cos(x^2)) = {analytical.item():.6f}")
```

**출력:**

```
x = 1.0
y = sin(x^2) = 0.841471
dy/dx (autograd) = 1.080605
dy/dx (해석적: 2x*cos(x^2)) = 1.080605
```

#### Taylor Series

*테일러 급수는 함수의 다항식 근사를 제공합니다.*

```python
import torch
import math

# e^x의 테일러 급수 근사 (x=0 주변)
# e^x = 1 + x + x^2/2! + x^3/3! + ...
x = torch.tensor(1.0)
n_terms = 10

approx = torch.tensor(0.0)
for n in range(n_terms):
    approx += (x ** n) / math.factorial(n)

print(f"x = {x.item()}")
print(f"테일러 급수 (10항): {approx.item():.10f}")
print(f"torch.exp(x): {torch.exp(x).item():.10f}")
print(f"math.e: {math.e:.10f}")
```

**출력:**

```
x = 1.0
테일러 급수 (10항): 2.7182817459
torch.exp(x): 2.7182817459
math.e: 2.7182818285
```

---

## ∂ 미분방정식

- 1차 선형: dy/dx + P(x)y = Q(x)
- 적분인자: μ(x) = e^{∫P(x)dx}
- 분리형: dy/dx = g(x)h(y)
- 완전형: M(x,y)dx + N(x,y)dy = 0
- 2차 선형: y'' + p(x)y' + q(x)y = r(x)
- 특성방정식: ar² + br + c = 0
- 동차해: y_h = c₁e^{r₁x} + c₂e^{r₂x}
- 파동 방정식: ∂²u/∂t² = c²∇²u
- 열 방정식: ∂u/∂t = α∇²u
- 라플라스 방정식: ∇²u = 0
- 슈뢰딩거 방정식: iℏ∂ψ/∂t = Ĥψ
- 오일러-라그랑주: ∂L/∂q - d/dt(∂L/∂q̇) = 0

---

## ⊗ 선형대수

- 행렬곱: (AB)ᵢⱼ = ∑ₖ AᵢₖBₖⱼ
- 2×2 행렬식: |A| = ad - bc
- 3×3 행렬식: 사뤼스 법칙
- 역행렬: A⁻¹A = AA⁻¹ = I
- 전치: (Aᵀ)ᵢⱼ = Aⱼᵢ
- 고유값: Av = λv
- 특성다항식: det(A - λI) = 0
- 대각합: tr(A) = ∑ᵢ Aᵢᵢ = ∑λᵢ
- 계수: dim(Im(A))
- 영공간: ker(A) = {x : Ax = 0}
- 그람-슈미트: 직교화
- 특이값분해: A = UΣVᵀ
- QR 분해: A = QR

### 코드 예제

#### Matrix Operations

*행렬 곱셈은 (AB)_ij = sum_k A_ik * B_kj 규칙을 따릅니다.*

```python
import torch

# 행렬 연산
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])

print("행렬 A:")
print(A)
print("\n행렬 B:")
print(B)

# 행렬 곱: (AB)_ij = sum_k A_ik * B_kj
print("\nA @ B (행렬 곱):")
print(A @ B)

# 원소별 곱 (아다마르 곱)
print("\nA * B (원소별 곱):")
print(A * B)
```

**출력:**

```
행렬 A:
tensor([[1., 2.],
        [3., 4.]])

행렬 B:
tensor([[5., 6.],
        [7., 8.]])

A @ B (행렬 곱):
tensor([[19., 22.],
        [43., 50.]])

A * B (원소별 곱):
tensor([[ 5., 12.],
        [21., 32.]])
```

#### Determinant Inverse

*2x2 행렬 [[a,b],[c,d]]의 행렬식 = ad-bc. det != 0일 때만 역행렬 존재.*

```python
import torch

# 행렬식과 역행렬
A = torch.tensor([[1., 2.], [3., 4.]])

print("행렬 A:")
print(A)

# 행렬식: |A| = ad - bc
det = torch.det(A)
print(f"\n행렬식: {det.item():.4f}")
print(f"수동 계산: 1*4 - 2*3 = {1*4 - 2*3}")

# 역행렬: A^(-1) * A = I
A_inv = torch.inverse(A)
print("\n역행렬 A^(-1):")
print(A_inv)

print("\n검증 A @ A^(-1) = I:")
print(A @ A_inv)
```

**출력:**

```
행렬 A:
tensor([[1., 2.],
        [3., 4.]])

행렬식: -2.0000
수동 계산: 1*4 - 2*3 = -2

역행렬 A^(-1):
tensor([[-2.0000,  1.0000],
        [ 1.5000, -0.5000]])

검증 A @ A^(-1) = I:
tensor([[1., 0.],
        [0., 1.]])
```

#### Eigenvalues

*고유값은 det(A - lambda*I) = 0을 만족. 고유벡터는 Av = lambda*v를 만족.*

```python
import torch

# 고유값과 고유벡터: Av = lambda * v
A = torch.tensor([[4., 2.], [1., 3.]])

print("행렬 A:")
print(A)

# 고유값 계산
eigenvalues = torch.linalg.eigvalsh(A.float())
print(f"\n고유값: {eigenvalues}")

# 전체 분해
eigvals, eigvecs = torch.linalg.eig(A)
print(f"\n고유값 (복소수): {eigvals}")
print("고유벡터:")
print(eigvecs)
```

**출력:**

```
행렬 A:
tensor([[4., 2.],
        [1., 3.]])

고유값: tensor([2.3820, 4.6180])

고유값 (복소수): tensor([5.+0.j, 2.+0.j])
고유벡터:
tensor([[ 0.8944+0.j, -0.7071+0.j],
        [ 0.4472+0.j,  0.7071+0.j]])
```

#### Svd Decomposition

*SVD는 모든 행렬을 직교 행렬 U, V와 특이값 S로 분해합니다.*

```python
import torch

# 특이값 분해: A = U * Sigma * V^T
A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

print("행렬 A:")
print(A)
print(f"크기: {A.shape}")

U, S, Vh = torch.linalg.svd(A)
print(f"\nU 크기: {U.shape}")
print(f"S (특이값): {S}")
print(f"Vh 크기: {Vh.shape}")

# 재구성
reconstructed = U @ torch.diag(S) @ Vh[:S.shape[0], :]
print("\n재구성된 A:")
print(reconstructed)
```

**출력:**

```
행렬 A:
tensor([[1., 2., 3.],
        [4., 5., 6.]])
크기: torch.Size([2, 3])

U 크기: torch.Size([2, 2])
S (특이값): tensor([9.5080, 0.7729])
Vh 크기: torch.Size([3, 3])

재구성된 A:
tensor([[1.0000, 2.0000, 3.0000],
        [4.0000, 5.0000, 6.0000]])
```

---

## ∇ 벡터 해석

- 기울기: ∇f = (∂f/∂x)î + (∂f/∂y)ĵ + (∂f/∂z)k̂
- 발산: ∇·F = ∂Fₓ/∂x + ∂Fᵧ/∂y + ∂Fᵤ/∂z
- 회전: ∇×F = |î  ĵ  k̂|
-         |∂/∂x ∂/∂y ∂/∂z|
-         |Fₓ  Fᵧ  Fᵤ|
- 선적분: ∫_C F·dr
- 면적분: ∬_S F·n̂ dS
- 그린 정리: ∮_C F·dr = ∬_D (∂Q/∂x - ∂P/∂y)dA
- 스토크스 정리: ∮_C F·dr = ∬_S (∇×F)·n̂ dS
- 발산 정리: ∬_S F·n̂ dS = ∭_V ∇·F dV
- 방향 도함수: D_û f = ∇f · û

---

## ℂ 복소해석

- 오일러 공식: e^{iθ} = cos θ + i sin θ
- 켤레 복소수: z̄ = a - bi
- 절대값: |z| = √(a² + b²)
- 편각: arg(z) = arctan(b/a)
- 드 무아브르: (cos θ + i sin θ)ⁿ = cos(nθ) + i sin(nθ)
- 코시-리만: ∂u/∂x = ∂v/∂y, ∂u/∂y = -∂v/∂x
- 코시 적분: f(z₀) = 1/(2πi) ∮_C f(z)/(z-z₀) dz
- 유수 정리: ∮_C f(z)dz = 2πi ∑Res(f, zₖ)
- 로랑 급수: f(z) = ∑_{n=-∞}^∞ aₙ(z-z₀)ⁿ
- 해석적 연속
- 리만 곡면

### 코드 예제

#### Euler Formula

*오일러 공식은 복소평면에서 지수함수와 삼각함수를 연결합니다.*

```python
import torch
import math

# 오일러 공식: e^(i*theta) = cos(theta) + i*sin(theta)
theta = torch.tensor(math.pi / 4)  # 45도

# torch 복소 지수 사용
z = torch.exp(1j * theta)
print(f"theta = pi/4 = {theta.item():.6f}")
print(f"\ne^(i*theta) = {z}")
print(f"실수부: {z.real.item():.6f}")
print(f"허수부: {z.imag.item():.6f}")

# cos, sin으로 검증
print(f"\ncos(theta) = {math.cos(theta.item()):.6f}")
print(f"sin(theta) = {math.sin(theta.item()):.6f}")

# 유명한 항등식: e^(i*pi) + 1 = 0
euler_identity = torch.exp(1j * torch.tensor(math.pi)) + 1
print(f"\n오일러 항등식: e^(i*pi) + 1 = {euler_identity}")
print(f"(약 0이어야 함)")
```

**출력:**

```
theta = pi/4 = 0.785398

e^(i*theta) = (0.7071067690849304+0.7071067690849304j)
실수부: 0.707107
허수부: 0.707107

cos(theta) = 0.707107
sin(theta) = 0.707107

오일러 항등식: e^(i*pi) + 1 = -8.742277657347586e-08j
(약 0이어야 함)
```

---

## μ 통계 & 확률

- 평균: μ = (1/n)∑xᵢ
- 분산: σ² = E[(X - μ)²]
- 표준편차: σ = √(σ²)
- 공분산: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
- 상관계수: ρ = Cov(X,Y)/(σₓσᵧ)
- 베이즈 정리: P(A|B) = P(B|A)P(A)/P(B)
- 정규분포: f(x) = (1/σ√(2π))e^{-(x-μ)²/(2σ²)}
- 이항분포: P(X=k) = C(n,k)p^k(1-p)^{n-k}
- 포아송분포: P(X=k) = (λ^k e^{-λ})/k!
- 중심극한정리: X̄ ~ N(μ, σ²/n)
- 카이제곱: χ² = ∑(Oᵢ - Eᵢ)²/Eᵢ
- t-분포: t = (X̄ - μ)/(s/√n)

### 코드 예제

#### Mean Variance

*분산은 평균 주변의 데이터 분산을 측정합니다.*

```python
import torch

# 평균과 분산
data = torch.tensor([2., 4., 4., 4., 5., 5., 7., 9.])

# 평균: mu = (1/n) * sum(x_i)
mean = data.mean()
print(f"데이터: {data.tolist()}")
print(f"평균 (μ): {mean.item():.4f}")

# 분산: sigma^2 = E[(X - mu)^2]
variance = data.var(unbiased=False)  # 모분산
print(f"분산 (σ²): {variance.item():.4f}")

# 표준편차: sigma = sqrt(분산)
std = data.std(unbiased=False)
print(f"표준편차 (σ): {std.item():.4f}")

# 수동 계산
manual_var = ((data - mean) ** 2).mean()
print(f"\n수동 분산 계산: {manual_var.item():.4f}")
```

**출력:**

```
데이터: [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
평균 (μ): 5.0000
분산 (σ²): 4.0000
표준편차 (σ): 2.0000

수동 분산 계산: 4.0000
```

#### Covariance Correlation

*상관계수는 공분산을 [-1, 1] 범위로 정규화합니다.*

```python
import torch

# 공분산과 상관계수
X = torch.tensor([1., 2., 3., 4., 5.])
Y = torch.tensor([2., 4., 5., 4., 5.])

# 공분산: Cov(X,Y) = E[(X-mu_x)(Y-mu_y)]
mean_x, mean_y = X.mean(), Y.mean()
cov = ((X - mean_x) * (Y - mean_y)).mean()

print(f"X: {X.tolist()}")
print(f"Y: {Y.tolist()}")
print(f"공분산: {cov.item():.4f}")

# 상관계수: rho = Cov(X,Y) / (sigma_x * sigma_y)
std_x = X.std(unbiased=False)
std_y = Y.std(unbiased=False)
correlation = cov / (std_x * std_y)
print(f"상관계수: {correlation.item():.4f}")
```

**출력:**

```
X: [1.0, 2.0, 3.0, 4.0, 5.0]
Y: [2.0, 4.0, 5.0, 4.0, 5.0]
공분산: 1.2000
상관계수: 0.7746
```

#### Normal Distribution

*정규분포는 통계와 머신러닝에서 기본이 되는 분포입니다.*

```python
import torch
import math

# 정규 (가우시안) 분포
# f(x) = (1/(sigma*sqrt(2*pi))) * exp(-(x-mu)^2/(2*sigma^2))
mu = 0.0
sigma = 1.0
x = torch.linspace(-3, 3, 7)

def normal_pdf(x, mu, sigma):
    coef = 1 / (sigma * math.sqrt(2 * math.pi))
    exp_term = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coef * exp_term

pdf = normal_pdf(x, mu, sigma)
print(f"x 값: {x.tolist()}")
print(f"확률밀도함수 값:")
for xi, pi in zip(x.tolist(), pdf.tolist()):
    print(f"  f({xi:5.2f}) = {pi:.6f}")
```

**출력:**

```
x 값: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
확률밀도함수 값:
  f(-3.00) = 0.004432
  f(-2.00) = 0.053991
  f(-1.00) = 0.241971
  f( 0.00) = 0.398942
  f( 1.00) = 0.241971
  f( 2.00) = 0.053991
  f( 3.00) = 0.004432
```

#### Bayes Theorem

*베이즈 정리는 새로운 증거가 주어졌을 때 확률을 업데이트합니다.*

```python
import torch

# 베이즈 정리: P(A|B) = P(B|A) * P(A) / P(B)
# 예: 의료 검사
# P(질병) = 0.01 (1%가 질병 보유)
# P(양성|질병) = 0.99 (99% 참양성률)
# P(양성|정상) = 0.05 (5% 위양성률)

P_disease = torch.tensor(0.01)
P_positive_given_disease = torch.tensor(0.99)
P_positive_given_no_disease = torch.tensor(0.05)

# P(양성) = P(+|D)*P(D) + P(+|~D)*P(~D)
P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_no_disease * (1 - P_disease))

# P(질병|양성) = P(+|D) * P(D) / P(+)
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print("베이즈 정리 예시: 의료 검사")
print(f"사전확률 P(질병): {P_disease.item():.2%}")
print(f"P(양성|질병): {P_positive_given_disease.item():.2%}")
print(f"P(양성|정상): {P_positive_given_no_disease.item():.2%}")
print(f"\nP(양성): {P_positive.item():.4f}")
print(f"P(질병|양성): {P_disease_given_positive.item():.2%}")
print("\n양성 판정을 받아도 실제 질병 확률은 ~17%!")
```

**출력:**

```
베이즈 정리 예시: 의료 검사
사전확률 P(질병): 1.00%
P(양성|질병): 99.00%
P(양성|정상): 5.00%

P(양성): 0.0594
P(질병|양성): 16.67%

양성 판정을 받아도 실제 질병 확률은 ~17%!
```

---

## Σ 급수 & 수열

- 등차수열: aₙ = a₁ + (n-1)d
- 등비수열: aₙ = a₁ · r^{n-1}
- 등차급수: Sₙ = n(a₁ + aₙ)/2
- 등비급수: Sₙ = a₁(1 - rⁿ)/(1 - r)
- 무한등비급수: S = a₁/(1 - r), |r| < 1
- 조화급수: ∑(1/n) 발산
- p-급수: ∑(1/n^p) p > 1이면 수렴
- 교대급수: ∑(-1)ⁿaₙ
- 비율판정법: lim|aₙ₊₁/aₙ| < 1 → 수렴
- 거듭제곱근판정법: lim|aₙ|^{1/n} < 1 → 수렴
- 적분판정법: ∫f(x)dx 수렴 ↔ ∑f(n) 수렴
- 비교판정법

---

## ∿ 삼각함수

- 피타고라스: sin²θ + cos²θ = 1
- tan²θ + 1 = sec²θ
- 1 + cot²θ = csc²θ
- 배각공식: sin(2θ) = 2sin(θ)cos(θ)
- cos(2θ) = cos²θ - sin²θ = 2cos²θ - 1
- 반각공식: sin²(θ/2) = (1 - cos θ)/2
- 합/차 공식: sin(α ± β) = sin α cos β ± cos α sin β
- cos(α ± β) = cos α cos β ∓ sin α sin β
- 곱→합 공식: sin A sin B = ½[cos(A-B) - cos(A+B)]
- 사인 법칙: a/sin A = b/sin B = c/sin C
- 코사인 법칙: c² = a² + b² - 2ab cos C
- 역함수: sin⁻¹x + cos⁻¹x = π/2

### 코드 예제

#### Trig Identities

*삼각함수 항등식은 수식을 단순화하는 데 기본이 됩니다.*

```python
import torch
import math

theta = torch.tensor(math.pi / 6)  # 30도

print(f"theta = pi/6 (30도)")
print()

# 피타고라스 항등식: sin^2 + cos^2 = 1
sin_t = torch.sin(theta)
cos_t = torch.cos(theta)
identity = sin_t**2 + cos_t**2
print(f"sin^2(theta) + cos^2(theta) = {identity.item():.10f}")

# 배각 공식
sin_2t = torch.sin(2 * theta)
double_angle = 2 * sin_t * cos_t
print(f"\nsin(2*theta) = {sin_2t.item():.6f}")
print(f"2*sin(theta)*cos(theta) = {double_angle.item():.6f}")

# cos(2*theta) = cos^2 - sin^2 검증
cos_2t = torch.cos(2 * theta)
cos_double = cos_t**2 - sin_t**2
print(f"\ncos(2*theta) = {cos_2t.item():.6f}")
print(f"cos^2(theta) - sin^2(theta) = {cos_double.item():.6f}")
```

**출력:**

```
theta = pi/6 (30도)

sin^2(theta) + cos^2(theta) = 1.0000000000

sin(2*theta) = 0.866025
2*sin(theta)*cos(theta) = 0.866025

cos(2*theta) = 0.500000
cos^2(theta) - sin^2(theta) = 0.500000
```

---

## ∮ 정수론

- 소수 정리: π(x) ~ x/ln(x)
- 오일러 파이 함수: φ(n) = n∏(1 - 1/p)
- 페르마 소정리: a^{p-1} ≡ 1 (mod p)
- 오일러 정리: a^{φ(n)} ≡ 1 (mod n)
- 중국인의 나머지 정리
- 이차 상호법칙: (p/q)(q/p) = (-1)^{(p-1)(q-1)/4}
- 약수 함수: σ(n) = ∑_{d|n} d
- 뫼비우스 함수: μ(n)
- 리만 제타: ζ(s) = ∑(1/n^s)
- 골드바흐 추측
- 쌍둥이 소수 추측

---

## ∫ 푸리에 해석

- 푸리에 급수: f(x) = a₀/2 + ∑[aₙcos(nx) + bₙsin(nx)]
- 계수: aₙ = (2/π)∫f(x)cos(nx)dx
- bₙ = (2/π)∫f(x)sin(nx)dx
- 복소 형태: f(x) = ∑cₙe^{inx}
- 푸리에 변환: F(ω) = ∫f(t)e^{-iωt}dt
- 역변환: f(t) = (1/2π)∫F(ω)e^{iωt}dω
- 컨볼루션: (f * g)(t) = ∫f(τ)g(t-τ)dτ
- 파세발 정리: ∫|f(x)|²dx = (1/2π)∫|F(ω)|²dω
- 이산 푸리에 변환: Xₖ = ∑xₙe^{-2πikn/N}
- 고속 푸리에 변환: O(N log N) 알고리즘

### 코드 예제

#### Fourier Transform

*FFT는 신호를 O(N log N) 시간에 주파수 성분으로 분해합니다.*

```python
import torch
import math

# 이산 푸리에 변환 (DFT)
# X_k = sum_n x_n * e^(-2*pi*i*k*n/N)

# 간단한 신호 생성: 두 정현파의 합
N = 64
t = torch.linspace(0, 1, N)
freq1, freq2 = 5, 12  # Hz
signal = torch.sin(2 * math.pi * freq1 * t) + 0.5 * torch.sin(2 * math.pi * freq2 * t)

# FFT 계산
fft_result = torch.fft.fft(signal)
magnitudes = torch.abs(fft_result)
freqs = torch.fft.fftfreq(N, 1/N)

# 피크 찾기 (상위 주파수)
top_idx = torch.argsort(magnitudes[:N//2], descending=True)[:3]
print("신호: sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)")
print(f"\n주요 주파수 성분:")
for idx in top_idx:
    print(f"  주파수: {freqs[idx].item():.1f} Hz, 크기: {magnitudes[idx].item():.2f}")
```

**출력:**

```
신호: sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)

주요 주파수 성분:
  주파수: 5.0 Hz, 크기: 31.63
  주파수: 12.0 Hz, 크기: 14.37
  주파수: 13.0 Hz, 크기: 4.18
```

---

## Γ 특수 함수

- 감마 함수: Γ(n) = (n-1)! = ∫₀^∞ t^{n-1}e^{-t}dt
- 베타 함수: B(x,y) = ∫₀¹ t^{x-1}(1-t)^{y-1}dt
- 오차 함수: erf(x) = (2/√π)∫₀ˣ e^{-t²}dt
- 베셀 함수: Jₙ(x)
- 르장드르 다항식: Pₙ(x)
- 에르미트 다항식: Hₙ(x)
- 라게르 다항식: Lₙ(x)
- 체비셰프 다항식: Tₙ(x)
- 초기하 함수: ₂F₁(a,b;c;z)
- 타원 적분: K(k), E(k)
- 리만 제타: ζ(s) = ∑(1/n^s)

### 코드 예제

#### Gamma Function

*감마 함수는 팩토리얼을 비정수로 확장합니다.*

```python
import torch
import math

# 감마 함수: Gamma(n) = (n-1)! (양의 정수)
# Gamma(n) = integral_0^inf t^(n-1) * e^(-t) dt

# 정수의 경우, Gamma(n) = (n-1)!
print("감마 함수: Gamma(n) = (n-1)!")
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

**출력:**

```
감마 함수: Gamma(n) = (n-1)!

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

## ⊕ 최적화

- 경사하강법: xₙ₊₁ = xₙ - α∇f(xₙ)
- 뉴턴 방법: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
- 라그랑주 승수법: ∇f = λ∇g
- KKT 조건: 정상성, 원시/쌍대 가능성
- 볼록 최적화: f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
- 선형 계획법: max cᵀx s.t. Ax ≤ b
- 이차 계획법: min ½xᵀQx + cᵀx
- 켤레 기울기법
- BFGS 알고리즘
- 내점법
- 심플렉스 알고리즘

### 코드 예제

#### Gradient Descent

*경사하강법은 음의 기울기 방향을 따라 반복적으로 최솟값으로 이동합니다.*

```python
import torch

# 경사하강법: x_(n+1) = x_n - alpha * grad f(x_n)
# f(x) = x^2 - 4x + 4 = (x-2)^2 최소화
# 최솟값: x = 2

x = torch.tensor(0.0, requires_grad=True)
lr = 0.1  # 학습률 (alpha)

print("경사하강법으로 f(x) = (x-2)^2 최소화")
print(f"시작점 x = {x.item():.4f}")
print()

for i in range(10):
    # 순전파: f(x) 계산
    y = (x - 2) ** 2

    # 역전파: 기울기 계산
    y.backward()

    # 업데이트: x = x - lr * grad
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()

    if i % 2 == 0:
        print(f"단계 {i}: x = {x.item():.4f}, f(x) = {((x-2)**2).item():.6f}")

print(f"\n최종 x = {x.item():.4f} (최적값 = 2.0)")
```

**출력:**

```
경사하강법으로 f(x) = (x-2)^2 최소화
시작점 x = 0.0000

단계 0: x = 0.4000, f(x) = 2.560000
단계 2: x = 0.9760, f(x) = 1.048576
단계 4: x = 1.3446, f(x) = 0.429497
단계 6: x = 1.5806, f(x) = 0.175922
단계 8: x = 1.7316, f(x) = 0.072058

최종 x = 1.7853 (최적값 = 2.0)
```

#### Newtons Method

*뉴턴 방법은 잘 정의된 함수에서 이차 수렴합니다.*

```python
import torch

# 뉴턴 방법: x_(n+1) = x_n - f(x_n) / f'(x_n)
# f(x) = x^2 - 2의 근 찾기 (즉, sqrt(2))

x = torch.tensor(1.0, requires_grad=True)
print("뉴턴 방법으로 sqrt(2) 찾기")
print(f"x^2 - 2 = 0 풀기")
print(f"시작점 x = {x.item():.4f}")
print()

for i in range(6):
    # f(x) = x^2 - 2
    y = x ** 2 - 2

    # f'(x) = 2x
    y.backward()
    f_prime = x.grad.item()

    # 뉴턴 업데이트: x = x - f(x)/f'(x)
    with torch.no_grad():
        x -= y / f_prime
        x.grad.zero_()

    print(f"단계 {i}: x = {x.item():.10f}")

print(f"\n결과: {x.item():.10f}")
print(f"실제 sqrt(2): {2**0.5:.10f}")
```

**출력:**

```
뉴턴 방법으로 sqrt(2) 찾기
x^2 - 2 = 0 풀기
시작점 x = 1.0000

단계 0: x = 1.5000000000
단계 1: x = 1.4166666269
단계 2: x = 1.4142156839
단계 3: x = 1.4142135382
단계 4: x = 1.4142135382
단계 5: x = 1.4142135382

결과: 1.4142135382
실제 sqrt(2): 1.4142135624
```

---

## ⚛ 물리 공식

- 뉴턴 제2법칙: F = ma = dp/dt
- 에너지: E = mc²
- 운동량: p = mv
- 각운동량: L = r × p
- 슈뢰딩거 방정식: iℏ∂ψ/∂t = Ĥψ
- 맥스웰 방정식:
-   ∇·E = ρ/ε₀
-   ∇·B = 0
-   ∇×E = -∂B/∂t
-   ∇×B = μ₀(J + ε₀∂E/∂t)
- 로렌츠 변환: x' = γ(x - vt)
- 플랑크 법칙: E = hν
- 하이젠베르크: ΔxΔp ≥ ℏ/2

### 코드 예제

#### Wave Equation

*파동 방정식은 매질을 통한 파동 전파를 설명합니다.*

```python
import torch

# 파동 방정식: d2u/dt2 = c^2 * d2u/dx2
# 유한 차분을 사용한 간단한 수치 해

# 매개변수
nx, nt = 50, 100
dx, dt = 0.1, 0.01
c = 1.0  # 파동 속도
r = (c * dt / dx) ** 2

# 초기 조건: 가우시안 펄스
x = torch.linspace(0, (nx-1)*dx, nx)
u = torch.exp(-((x - 2.5) ** 2))

# 해 저장
u_prev = u.clone()
u_curr = u.clone()

print("파동 방정식 시뮬레이션")
print(f"파동 속도 c = {c}")
print(f"격자: {nx}점, {nt} 시간 단계")
print(f"\nx=2.5에서 초기 펄스")
print(f"t=0에서 최대 진폭: {u_curr.max().item():.4f}")

# 시간 전진
for t in range(nt):
    u_next = torch.zeros_like(u_curr)
    for i in range(1, nx-1):
        u_next[i] = 2*u_curr[i] - u_prev[i] + r*(u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
    u_prev = u_curr.clone()
    u_curr = u_next.clone()

    if t == nt//2:
        print(f"t={t}에서 최대 진폭: {u_curr.max().item():.4f}")

print(f"t={nt}에서 최대 진폭: {u_curr.max().item():.4f}")
print("\n파동이 시간에 따라 전파되고 퍼집니다.")
```

**출력:**

```
파동 방정식 시뮬레이션
파동 속도 c = 1.0
격자: 50점, 100 시간 단계

x=2.5에서 초기 펄스
t=0에서 최대 진폭: 1.0000
t=50에서 최대 진폭: 0.7679
t=100에서 최대 진폭: 0.5088

파동이 시간에 따라 전파되고 퍼집니다.
```

---
