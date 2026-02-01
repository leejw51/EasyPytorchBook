# 쉬운 트랜스포머 v1.0

**신경망 수학 & 트랜스포머 아키텍처**

*by jw*

이 책은 트랜스포머 아키텍처와 신경망 수학을 다룹니다:

- **어텐션 메커니즘** - 셀프 어텐션, 멀티헤드 어텐션
- **역전파** - 트랜스포머를 통한 기울기 흐름
- **학습 역학** - 최적화 및 정규화
- **코드 예제** - PyTorch 구현

*PyTorch 2.8.0 사용*

---

## 목차

1. [트랜스포머 아키텍처 흐름](#트랜스포머-아키텍처-흐름)
2. [셀프 어텐션: 핵심](#셀프-어텐션-핵심)
3. [Softmax + CrossEntropy 마법](#Softmax--CrossEntropy-마법)
4. [활성화 함수: 비선형성의 힘](#활성화-함수-비선형성의-힘)
5. [피드포워드 네트워크 (FFN)](#피드포워드-네트워크-(FFN))
6. [레이어 정규화 흐름](#레이어-정규화-흐름)
7. [위치 인코딩](#위치-인코딩)
8. [기울기 흐름 & 역전파](#기울기-흐름--역전파)
9. [학습 역학](#학습-역학)
10. [어텐션 패턴](#어텐션-패턴)
11. [임베딩 & 표현](#임베딩--표현)
12. [마스크드 언어 모델링](#마스크드-언어-모델링)
13. [자기회귀 생성](#자기회귀-생성)
14. [스케일 & 창발](#스케일--창발)
15. [최적화 기법](#최적화-기법)
16. [현대적 개선](#현대적-개선)

---

## 🔄 트랜스포머 아키텍처 흐름

- 입력 흐름: 텍스트 -> 토큰 -> 임베딩 + 위치
- 1. 토큰화: 'Hello' -> [H, e, l, l, o] -> [15496, 8894, 75, 75, 78]
- 2. 임베딩: token_id -> E[token_id] ∈ ℝᵈ
- 3. 위치: PE(pos,2i) = sin(pos/10000^(2i/d))
- 4. 입력: x = 임베딩 + 위치인코딩
- 5. N × 트랜스포머블록: x -> 멀티헤드어텐션 -> FFN
- 6. 출력: 최종 레이어 -> Softmax -> 확률
- 흐름: 입력 -> 임베드 -> N×[어텐션+FFN] -> 출력
- 각 블록: x -> Attn(x) + x -> FFN(x) + x
- 잔차 연결: 기울기 소실 방지
- 레이어 정규화: 각 단계에서 학습 안정화

### 코드 예제

#### Transformer Block

*트랜스포머 블록은 셀프 어텐션, 피드포워드 네트워크, 레이어 정규화, 잔차 연결을 결합합니다.*

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    '''단일 트랜스포머 블록: 어텐션 + FFN + 잔차 + LayerNorm'''
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
        # 잔차가 있는 셀프 어텐션
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        # 잔차가 있는 FFN
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ln2(x)

        return x

# 생성 및 테스트
block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
x = torch.randn(2, 10, 64)  # batch=2, seq=10, d_model=64

output = block(x)
print(f"입력 형태:  {x.shape}")
print(f"출력 형태: {output.shape}")

n_params = sum(p.numel() for p in block.parameters())
print(f"\n파라미터: {n_params:,}")
print("\n블록 = 어텐션 + LayerNorm + FFN + LayerNorm + 잔차")
```

**출력:**

```
입력 형태:  torch.Size([2, 10, 64])
출력 형태: torch.Size([2, 10, 64])

파라미터: 49,984

블록 = 어텐션 + LayerNorm + FFN + LayerNorm + 잔차
```

---

## 🎯 셀프 어텐션: 핵심

- 마법의 공식: Attn(Q,K,V) = softmax(QK^T/√d_k)V
- 단계 1: Q=XW_Q, K=XW_K, V=XW_V (선형 투영)
- 단계 2: 점수 = QK^T/√d_k (스케일 내적)
- 단계 3: 가중치 = softmax(점수) (어텐션 맵)
- 단계 4: 출력 = 가중치 × V (가중 합)
- 왜 √d_k?: softmax 포화 방지
- 어텐션 가중치: Σⱼ αᵢⱼ = 1 (확률 분포)
- 멀티헤드: h개 헤드 병렬 실행, 후 연결
- MHA = Concat(head₁,...,head_h)W_O
- 각 헤드는 다른 관계를 학습
- 인과적 마스크: 미래 토큰 참조 방지

### 코드 예제

#### Self Attention

*셀프 어텐션은 쿼리-키 유사도를 기반으로 값의 가중 조합을 계산합니다.*

```python
import torch
import torch.nn.functional as F
import math

# 셀프 어텐션: Attn(Q,K,V) = softmax(QK^T/sqrt(d_k))V
batch_size, seq_len, d_model = 2, 4, 8
d_k = d_model

# 입력 시퀀스
X = torch.randn(batch_size, seq_len, d_model)

# 선형 투영 (데모를 위해 간소화)
W_Q = torch.randn(d_model, d_k)
W_K = torch.randn(d_model, d_k)
W_V = torch.randn(d_model, d_k)

# 단계 1: Q, K, V 계산
Q = X @ W_Q  # (batch, seq, d_k)
K = X @ W_K
V = X @ W_V

print(f"X 형태: {X.shape}")
print(f"Q, K, V 형태: {Q.shape}")

# 단계 2: 스케일 내적 어텐션 점수
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
print(f"\n점수 형태: {scores.shape}")
print(f"점수[0]:\n{scores[0]}")

# 단계 3: Softmax로 어텐션 가중치 계산
attn_weights = F.softmax(scores, dim=-1)
print(f"\n어텐션 가중치 (행 합이 1):")
print(f"{attn_weights[0]}")
print(f"행 합: {attn_weights[0].sum(dim=-1)}")

# 단계 4: 값의 가중 합
output = attn_weights @ V
print(f"\n출력 형태: {output.shape}")
```

**출력:**

```
X 형태: torch.Size([2, 4, 8])
Q, K, V 형태: torch.Size([2, 4, 8])

점수 형태: torch.Size([2, 4, 4])
점수[0]:
tensor([[  6.5752,  -0.7889,  14.6079,  27.1378],
        [ -3.8368,   0.7991,  -5.4737, -11.2404],
        [  3.4245,  -0.5925,   9.8516,  21.3384],
        [ -1.3650,   0.1074,   4.8432,  13.4147]])

어텐션 가중치 (행 합이 1):
tensor([[1.1742e-09, 7.4398e-13, 3.6167e-06, 1.0000e+00],
        [9.5864e-03, 9.8854e-01, 1.8653e-03, 5.8384e-06],
        [1.6598e-08, 2.9891e-10, 1.0265e-05, 9.9999e-01],
        [3.8122e-07, 1.6620e-06, 1.8940e-04, 9.9981e-01]])
행 합: tensor([1.0000, 1.0000, 1.0000, 1.0000])

출력 형태: torch.Size([2, 4, 8])
```

#### Multi Head Attention

*멀티헤드 어텐션은 여러 어텐션 연산을 병렬로 실행하여 각각 다른 패턴을 학습합니다.*

```python
import torch
import torch.nn as nn

# 멀티헤드 어텐션
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

        # 선형 투영 및 헤드로 분할
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 어텐션 점수
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # 헤드 결합
        out = (attn @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(out)

# 테스트
mha = SimpleMultiHeadAttention(d_model=64, num_heads=4)
x = torch.randn(2, 8, 64)  # batch=2, seq=8, d_model=64
out = mha(x)
print(f"입력 형태: {x.shape}")
print(f"출력 형태: {out.shape}")
print(f"파라미터 수: {sum(p.numel() for p in mha.parameters())}")
```

**출력:**

```
입력 형태: torch.Size([2, 8, 64])
출력 형태: torch.Size([2, 8, 64])
파라미터 수: 16640
```

---

## 📊 Softmax + CrossEntropy 마법

- 이 조합이 기울기에 훌륭한 이유:
- Softmax: p_i = e^(z_i)/Σⱼe^(z_j) (확률로 변환)
- CrossEntropy: L = -Σᵢ y_i log(p_i) (오차 측정)
- 결합 기울기: ∂L/∂z_i = p_i - y_i
- ★ 이것은 놀랍도록 간단합니다! 복잡한 미분 없음
- 순전파: z -> softmax -> p -> CE -> 손실
- 역전파: ∂L/∂z = p - y (한 단계!)
- Softmax 미분: ∂p_i/∂z_j = p_i(δᵢⱼ - p_j)
- CE 미분: ∂L/∂p_i = -y_i/p_i
- 결합: 상쇄로 기울기가 깔끔해짐
- 이것이 트랜스포머가 잘 학습되는 이유!

### 코드 예제

#### Softmax Crossentropy

*Softmax + CrossEntropy는 아름답게 간단한 기울기를 가집니다: 확률에서 타겟을 빼면 됩니다.*

```python
import torch
import torch.nn.functional as F

# Softmax + CrossEntropy 기울기의 마법
# 결합 기울기: dL/dz = p - y (믿기 어렵게 간단!)

# 로짓 (모델의 원시 출력)
logits = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
# 실제 라벨 (원-핫)
target = torch.tensor([0])  # 클래스 0

print("=== 순전파 ===")
# Softmax: 확률로 변환
probs = F.softmax(logits, dim=-1)
print(f"로짓: {logits.detach()}")
print(f"Softmax 확률: {probs.detach()}")
print(f"확률 합: {probs.sum().item():.4f}")

# CrossEntropy 손실
loss = F.cross_entropy(logits, target)
print(f"\nCrossEntropy 손실: {loss.item():.4f}")

print("\n=== 역전파 ===")
loss.backward()

# 기울기는 간단히: p - y
# 클래스 0 (타겟): p[0] - 1
# 나머지: p[i] - 0
print(f"기울기 dL/dz: {logits.grad}")

# 검증: 기울기 = softmax_출력 - 원핫_타겟
one_hot = F.one_hot(target, num_classes=3).float()
expected_grad = probs.detach() - one_hot
print(f"예상값 (p - y): {expected_grad}")
print(f"\n기울기는 단순히 (p - y)! 복잡한 수학 불필요.")
```

**출력:**

```
=== 순전파 ===
로짓: tensor([[2.0000, 1.0000, 0.1000]])
Softmax 확률: tensor([[0.6590, 0.2424, 0.0986]])
확률 합: 1.0000

CrossEntropy 손실: 0.4170

=== 역전파 ===
기울기 dL/dz: tensor([[-0.3410,  0.2424,  0.0986]])
예상값 (p - y): tensor([[-0.3410,  0.2424,  0.0986]])

기울기는 단순히 (p - y)! 복잡한 수학 불필요.
```

---

## ⚡ 활성화 함수: 비선형성의 힘

- 왜 비선형? 선형 레이어는 복잡성 학습 불가
- ReLU: f(x) = max(0,x), f'(x) = x>0이면 1 아니면 0
- GELU: x·Φ(x) (부드러움, 트랜스포머에서 사용)
- Swish: x·σ(x) (부드러운 활성화)
- 비선형성 덕분에: 보편 근사 가능
- 활성화 없이: W₃W₂W₁x = W_결합·x
- 활성화 있으면: 어떤 함수도 근사 가능
- 기울기 흐름: 활성화가 기울기를 죽이면 안됨
- ReLU 문제: 죽은 뉴런 (기울기 = 0)
- GELU 해결: 항상 약간의 기울기
- 위치: 보통 FFN 레이어에서

### 코드 예제

#### Activations

*GELU와 Swish 같은 현대적 활성화 함수는 부드러운 기울기를 제공하여 죽은 뉴런 문제를 피합니다.*

```python
import torch
import torch.nn.functional as F

x = torch.linspace(-3, 3, 7)
print(f"입력 x: {x.tolist()}")
print()

# ReLU: max(0, x)
relu = F.relu(x)
print(f"ReLU:    {relu.tolist()}")

# GELU: x * Phi(x) - 트랜스포머에서 사용
gelu = F.gelu(x)
print(f"GELU:    {[f'{v:.3f}' for v in gelu.tolist()]}")

# Swish/SiLU: x * sigmoid(x)
swish = F.silu(x)
print(f"Swish:   {[f'{v:.3f}' for v in swish.tolist()]}")

# x=0에서 기울기 비교
x_grad = torch.tensor([0.0], requires_grad=True)
F.relu(x_grad).backward()
print(f"\nx=0에서 기울기:")
print(f"  ReLU:  {x_grad.grad.item()} (정의 안됨, PyTorch는 0 사용)")

x_grad = torch.tensor([0.0], requires_grad=True)
F.gelu(x_grad).backward()
print(f"  GELU:  {x_grad.grad.item():.4f} (부드러움, 0이 아님)")

x_grad = torch.tensor([0.0], requires_grad=True)
F.silu(x_grad).backward()
print(f"  Swish: {x_grad.grad.item():.4f} (부드러움, 0이 아님)")
```

**출력:**

```
입력 x: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

ReLU:    [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
GELU:    ['-0.004', '-0.046', '-0.159', '0.000', '0.841', '1.954', '2.996']
Swish:   ['-0.142', '-0.238', '-0.269', '0.000', '0.731', '1.762', '2.858']

x=0에서 기울기:
  ReLU:  0.0 (정의 안됨, PyTorch는 0 사용)
  GELU:  0.5000 (부드러움, 0이 아님)
  Swish: 0.5000 (부드러움, 0이 아님)
```

---

## 🧠 피드포워드 네트워크 (FFN)

- 구조: 선형 -> 활성화 -> 선형
- FFN(x) = W₂·activation(W₁x + b₁) + b₂
- 일반적: 4배 확장, 다시 d_model로
- 예시: 768 -> 3072 -> 768 (BERT)
- 목적: 비선형 변환 적용
- 각 위치가 독립적으로 처리됨
- 모델 용량과 비선형성 추가
- 기울기: 활성화를 통한 연쇄 법칙
- ∂L/∂W₁ = (∂L/∂h)·activation'(z₁)·x^T
- ∂L/∂W₂ = ∂L/∂y·h^T
- 대부분의 파라미터가 FFN 레이어에!

### 코드 예제

#### Ffn

*FFN은 각 위치에서 비선형 변환을 적용하며, 일반적으로 차원을 확장했다가 축소합니다.*

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 피드포워드 네트워크: 선형 -> 활성화 -> 선형
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x):
        # FFN(x) = W2 * activation(W1*x + b1) + b2
        return self.linear2(self.activation(self.linear1(x)))

# 일반적인 트랜스포머: 4배 확장
d_model = 768
d_ff = d_model * 4  # 3072

ffn = FFN(d_model, d_ff)
x = torch.randn(2, 10, d_model)  # batch=2, seq=10

output = ffn(x)
print(f"입력 형태: {x.shape}")
print(f"은닉 (확장): {d_ff}")
print(f"출력 형태: {output.shape}")

# 파라미터 수
n_params = sum(p.numel() for p in ffn.parameters())
print(f"\nFFN 파라미터: {n_params:,}")
print(f"  Linear1: {d_model} x {d_ff} + {d_ff} = {d_model * d_ff + d_ff:,}")
print(f"  Linear2: {d_ff} x {d_model} + {d_model} = {d_ff * d_model + d_model:,}")
print("\nFFN이 트랜스포머 파라미터의 대부분을 차지합니다!")
```

**출력:**

```
입력 형태: torch.Size([2, 10, 768])
은닉 (확장): 3072
출력 형태: torch.Size([2, 10, 768])

FFN 파라미터: 4,722,432
  Linear1: 768 x 3072 + 3072 = 2,362,368
  Linear2: 3072 x 768 + 768 = 2,360,064

FFN이 트랜스포머 파라미터의 대부분을 차지합니다!
```

---

## 📏 레이어 정규화 흐름

- 목적: 학습 안정화, 빠른 수렴
- LayerNorm: x̂ = (x - μ)/√(σ² + ε)
- μ = 평균(x), σ² = 분산(x) (특성 방향)
- 출력: y = γ·x̂ + β (학습 가능한 스케일/시프트)
- Pre-norm: LN(x) -> 어텐션 -> 잔차
- Post-norm: x -> 어텐션 -> LN -> 잔차
- 현대: Pre-norm이 더 안정적
- 기울기: ∂L/∂x는 평균과 분산 포함
- 정규화 -> 기울기 폭발 방지
- 각 레이어가 정규화된 입력을 받음
- 깊은 트랜스포머 학습에 필수

### 코드 예제

#### Layer Norm

*레이어 정규화는 특성 방향으로 정규화하여 학습을 안정화하고 더 깊은 네트워크를 가능하게 합니다.*

```python
import torch
import torch.nn as nn

# 레이어 정규화: 특성 방향으로 정규화
batch_size, seq_len, d_model = 2, 4, 8

x = torch.randn(batch_size, seq_len, d_model) * 10 + 5  # 정규화되지 않은 입력

print("=== LayerNorm 전 ===")
print(f"평균: {x.mean().item():.4f}")
print(f"표준편차: {x.std().item():.4f}")
print(f"샘플 값: {x[0, 0, :4].tolist()}")

# LayerNorm 적용
ln = nn.LayerNorm(d_model)
y = ln(x)

print("\n=== LayerNorm 후 ===")
print(f"평균: {y.mean().item():.4f} (0에 가까움)")
print(f"표준편차: {y.std().item():.4f} (1에 가까움)")
print(f"샘플 값: {[f'{v:.3f}' for v in y[0, 0, :4].tolist()]}")

# 수동 계산
print("\n=== 수동 LayerNorm ===")
x_sample = x[0, 0]  # 한 위치
mean = x_sample.mean()
var = x_sample.var(unbiased=False)
eps = 1e-5
x_norm = (x_sample - mean) / torch.sqrt(var + eps)
print(f"수동 평균: {x_norm.mean().item():.6f}")
print(f"수동 표준편차: {x_norm.std(unbiased=False).item():.6f}")
```

**출력:**

```
=== LayerNorm 전 ===
평균: 4.1201
표준편차: 10.8283
샘플 값: [20.94530487060547, 24.30544090270996, -5.878896713256836, -13.407829284667969]

=== LayerNorm 후 ===
평균: 0.0000 (0에 가까움)
표준편차: 1.0079 (1에 가까움)
샘플 값: ['1.326', '1.557', '-0.525', '-1.044']

=== 수동 LayerNorm ===
수동 평균: -0.000000
수동 표준편차: 1.000000
```

---

## 📍 위치 인코딩

- 문제: 어텐션은 위치 인식이 없음
- 해결: 입력에 위치 정보 추가
- 사인파: PE(pos,2i) = sin(pos/10000^(2i/d))
- PE(pos,2i+1) = cos(pos/10000^(2i/d))
- 속성: 각 위치마다 고유함
- 상대 위치: sin(a+b) = sin(a)cos(b)+...
- 학습된 PE: 학습 가능한 위치 임베딩
- 회전 PE: 복소수 회전 곱셈
- 위치 인코딩은 연결이 아닌 덧셈
- 길이 일반화 가능
- 다른 주파수가 다른 스케일 포착

### 코드 예제

#### Positional Encoding

*위치 인코딩은 다양한 주파수의 사인파를 사용하여 위치 정보를 추가합니다.*

```python
import torch
import math

def positional_encoding(max_len, d_model):
    '''사인파 위치 인코딩'''
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()

    # div_term = 10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
    pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스

    return pe

# 위치 인코딩 생성
max_len, d_model = 10, 8
pe = positional_encoding(max_len, d_model)

print(f"위치 인코딩 형태: {pe.shape}")
print(f"\n위치 0: {[f'{v:.3f}' for v in pe[0].tolist()]}")
print(f"위치 1: {[f'{v:.3f}' for v in pe[1].tolist()]}")
print(f"위치 9: {[f'{v:.3f}' for v in pe[9].tolist()]}")

# 각 위치가 고유한 인코딩을 가짐
print(f"\n모든 위치가 고유: {len(set(tuple(row.tolist()) for row in pe)) == max_len}")

# 저주파는 천천히, 고주파는 빠르게 변화
print(f"\n차원 0 (저주파) 위치별: {[f'{pe[i, 0].item():.2f}' for i in range(5)]}")
print(f"차원 6 (고주파) 위치별: {[f'{pe[i, 6].item():.2f}' for i in range(5)]}")
```

**출력:**

```
위치 인코딩 형태: torch.Size([10, 8])

위치 0: ['0.000', '1.000', '0.000', '1.000', '0.000', '1.000', '0.000', '1.000']
위치 1: ['0.841', '0.540', '0.100', '0.995', '0.010', '1.000', '0.001', '1.000']
위치 9: ['0.412', '-0.911', '0.783', '0.622', '0.090', '0.996', '0.009', '1.000']

모든 위치가 고유: True

차원 0 (저주파) 위치별: ['0.00', '0.84', '0.91', '0.14', '-0.76']
차원 6 (고주파) 위치별: ['0.00', '0.00', '0.00', '0.00', '0.00']
```

---

## ↩️ 기울기 흐름 & 역전파

- 트랜스포머 기울기 흐름은 복잡하지만 깔끔:
- 손실 -> Softmax -> 어텐션 -> 잔차 -> 입력
- 연쇄 법칙: ∂L/∂x = ∂L/∂y·∂y/∂x (합성)
- 잔차 연결: ∂L/∂x = ∂L/∂y·(1 + ∂F/∂x)
- 멀티헤드: 기울기가 헤드에 분산
- 어텐션 기울기: softmax 통과 (복잡)
- ∂αᵢⱼ/∂eᵢₖ = αᵢⱼ(δⱼₖ - αᵢₖ)
- 레이어 정규화: 평균/분산 기울기 포함
- FFN: 레이어 통한 표준 역전파
- 잔차는 기울기 소실 방지
- 레이어 정규화는 기울기 폭발 방지

### 코드 예제

#### Gradient Flow

*잔차 연결은 기울기가 직접 역전파될 수 있도록 하여 기울기 소실을 방지합니다.*

```python
import torch
import torch.nn as nn

# 잔차 연결을 통한 기울기 흐름 시연
class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 잔차: y = x + F(x)
        # 기울기: dy/dx = 1 + dF/dx (항상 최소 1!)
        return x + self.linear(x)

class NoResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x)

d_model = 64
num_layers = 10

# 잔차 연결 있음
residual_model = nn.Sequential(*[ResidualBlock(d_model) for _ in range(num_layers)])

# 잔차 연결 없음
no_residual_model = nn.Sequential(*[NoResidualBlock(d_model) for _ in range(num_layers)])

# 기울기 흐름 테스트
x = torch.randn(1, d_model, requires_grad=True)
y_res = residual_model(x).sum()
y_res.backward()
grad_res = x.grad.norm().item()

x = torch.randn(1, d_model, requires_grad=True)
y_no_res = no_residual_model(x).sum()
y_no_res.backward()
grad_no_res = x.grad.norm().item()

print(f"{num_layers}개 레이어를 통한 기울기 노름:")
print(f"  잔차 있음:    {grad_res:.4f}")
print(f"  잔차 없음:    {grad_no_res:.4f}")
print(f"\n잔차 연결이 기울기 흐름을 보존합니다!")
```

**출력:**

```
10개 레이어를 통한 기울기 노름:
  잔차 있음:    28.0388
  잔차 없음:    0.0360

잔차 연결이 기울기 흐름을 보존합니다!
```

---

## 🎓 학습 역학

- 손실 지형: 고차원, 비볼록
- Adam 옵티마이저: 파라미터별 적응 학습률
- m_t = β₁m_{t-1} + (1-β₁)∇L (모멘텀)
- v_t = β₂v_{t-1} + (1-β₂)∇L² (분산)
- θ_{t+1} = θ_t - α·m̂_t/√(v̂_t + ε)
- 학습률 스케줄: 워밍업 후 감쇠
- 워밍업: 초기 큰 업데이트 방지
- 가중치 감쇠: L_total = L + λ||θ||²
- 기울기 클리핑: 폭발 방지
- 배치 크기: 기울기 노이즈 영향
- 중요: 적절한 초기화 (Xavier/He)

### 코드 예제

#### Adam Optimizer

*Adam은 모멘텀과 적응적 학습률을 결합하여 효율적인 최적화를 수행합니다.*

```python
import torch
import torch.nn as nn

# Adam 옵티마이저: 적응적 학습률
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

print("Adam 옵티마이저 구성요소:")
print(f"  학습률 (alpha): {optimizer.defaults['lr']}")
print(f"  Beta1 (모멘텀): {optimizer.defaults['betas'][0]}")
print(f"  Beta2 (분산): {optimizer.defaults['betas'][1]}")
print(f"  Epsilon: {optimizer.defaults['eps']}")

# 학습 단계 시뮬레이션
x = torch.randn(32, 10)
target = torch.randn(32, 1)

print("\n학습 단계:")
for step in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()

    # 단계 전: 기울기 확인
    grad_norm = model.weight.grad.norm().item()

    optimizer.step()

    print(f"  단계 {step+1}: 손실={loss.item():.4f}, 기울기노름={grad_norm:.4f}")

print("\nAdam은 기울기 이력을 기반으로 파라미터별 학습률을 조정합니다!")
```

**출력:**

```
Adam 옵티마이저 구성요소:
  학습률 (alpha): 0.001
  Beta1 (모멘텀): 0.9
  Beta2 (분산): 0.999
  Epsilon: 1e-08

학습 단계:
  단계 1: 손실=1.3096, 기울기노름=1.4299
  단계 2: 손실=1.3057, 기울기노름=1.4254
  단계 3: 손실=1.3018, 기울기노름=1.4210
  단계 4: 손실=1.2979, 기울기노름=1.4165
  단계 5: 손실=1.2940, 기울기노름=1.4121

Adam은 기울기 이력을 기반으로 파라미터별 학습률을 조정합니다!
```

---

## 👁️ 어텐션 패턴

- 어텐션이 학습하는 것:
- 지역 패턴: 구문, 문법 규칙
- 장거리: 상호참조, 의미 관계
- 어텐션 헤드 전문화: 구문 vs 의미
- 초기 레이어: 지역 패턴 (품사, 구문)
- 후기 레이어: 전역 의미, 추론
- 어텐션 가중치: 해석 가능 (어느 정도)
- 인과적 마스킹: mask_{ij} = -∞ if i < j
- 셀프 어텐션: 각 토큰이 모든 토큰 참조
- 크로스 어텐션: 다른 시퀀스 참조
- 어텐션이 전부: 순환 없음!

### 코드 예제

#### Causal Mask

*인과적 마스킹은 생성 시 모델이 미래 토큰을 참조하는 것을 방지합니다.*

```python
import torch
import torch.nn.functional as F

# 디코더를 위한 인과적 (자기회귀) 마스킹
seq_len = 5

# 인과적 마스크 생성: 과거 위치만 참조 가능
# mask[i,j] = True는 위치 i가 위치 j를 참조할 수 있음을 의미
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
print("인과적 마스크 (1 = 참조 가능):")
print(causal_mask.int())

# 실제로는 참조 불가 위치에 -inf 사용
attn_mask = torch.zeros(seq_len, seq_len)
attn_mask = attn_mask.masked_fill(~causal_mask, float('-inf'))
print("\n어텐션 마스크 (softmax용):")
print(attn_mask)

# 예: 마스킹 전 어텐션 점수
scores = torch.randn(seq_len, seq_len)
print("\n원시 점수:")
print(scores.round(decimals=2))

# 마스크 적용 및 softmax
masked_scores = scores + attn_mask
attn_weights = F.softmax(masked_scores, dim=-1)
print("\n어텐션 가중치 (마스크 + softmax 후):")
print(attn_weights.round(decimals=2))
print("\n각 행이 현재와 과거 위치만 참조하는 것을 확인!")
```

**출력:**

```
인과적 마스크 (1 = 참조 가능):
tensor([[1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=torch.int32)

어텐션 마스크 (softmax용):
tensor([[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]])

원시 점수:
tensor([[-0.7200, -0.8100, -0.8400, -1.4800, -0.6800],
        [ 1.2300, -1.0500, -3.0200,  1.0000,  2.4900],
        [-1.3100,  1.4800,  1.5400, -0.0900, -0.9300],
        [-1.2500,  0.9100,  0.3800,  0.0300,  0.7800],
        [ 0.4600, -0.6300,  1.6000,  0.6100, -1.3700]])

어텐션 가중치 (마스크 + softmax 후):
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.9100, 0.0900, 0.0000, 0.0000, 0.0000],
        [0.0300, 0.4700, 0.5000, 0.0000, 0.0000],
        [0.0500, 0.4700, 0.2800, 0.2000, 0.0000],
        [0.1700, 0.0600, 0.5400, 0.2000, 0.0300]])

각 행이 현재와 과거 위치만 참조하는 것을 확인!
```

---

## 💎 임베딩 & 표현

- 토큰 임베딩: 이산 -> 연속
- 임베딩 행렬: V × d (어휘 × 차원)
- 조회: x_embed = E[token_id]
- 위치 인식: x = x_embed + x_pos
- 문맥적: 같은 단어, 다른 의미
- 깊은 표현: 계층적 특성
- 초기 레이어: 구문, 저수준 특성
- 깊은 레이어: 의미, 고수준 개념
- 표현 학습: 비지도
- 전이 학습: 사전훈련 -> 미세조정
- 임베딩 공간: 의미적 유사성

### 코드 예제

#### Embedding Lookup

*임베딩은 이산적인 토큰 ID를 모델이 처리할 수 있는 연속 벡터로 변환합니다.*

```python
import torch
import torch.nn as nn

# 토큰 임베딩: 이산 토큰 -> 연속 벡터
vocab_size = 1000
d_model = 64

embedding = nn.Embedding(vocab_size, d_model)

# 토큰 ID (예: 토크나이저에서)
token_ids = torch.tensor([[42, 123, 7, 500],
                          [99, 42, 200, 1]])  # batch=2, seq=4

print(f"토큰 ID 형태: {token_ids.shape}")
print(f"토큰 ID:\n{token_ids}")

# 임베딩 조회
embedded = embedding(token_ids)
print(f"\n임베딩 형태: {embedded.shape}")

# 같은 토큰 = 같은 임베딩
print(f"\n토큰 42가 두 번 등장. 같은 임베딩?")
print(f"  위치 (0,0): {embedded[0, 0, :4].tolist()}")
print(f"  위치 (1,1): {embedded[1, 1, :4].tolist()}")
print(f"  동일: {torch.equal(embedded[0, 0], embedded[1, 1])}")

# 임베딩 행렬
print(f"\n임베딩 행렬 형태: {embedding.weight.shape}")
print(f"총 파라미터: {vocab_size * d_model:,}")
```

**출력:**

```
토큰 ID 형태: torch.Size([2, 4])
토큰 ID:
tensor([[ 42, 123,   7, 500],
        [ 99,  42, 200,   1]])

임베딩 형태: torch.Size([2, 4, 64])

토큰 42가 두 번 등장. 같은 임베딩?
  위치 (0,0): [1.215755820274353, 0.5044317841529846, -0.6309076547622681, -1.3818917274475098]
  위치 (1,1): [1.215755820274353, 0.5044317841529846, -0.6309076547622681, -1.3818917274475098]
  동일: True

임베딩 행렬 형태: torch.Size([1000, 64])
총 파라미터: 64,000
```

---

## 🎭 마스크드 언어 모델링

- 사전훈련 목표: 마스크된 토큰 예측
- 입력: '고양이가 [MASK] 위에 앉았다'
- 타겟: [MASK]에 대해 '매트'를 예측
- 양방향: 왼쪽과 오른쪽 문맥 모두 사용
- 손실: 어휘에 대한 CrossEntropy
- 기울기: 전체 모델을 통해 역전파
- 자기지도: 사람 라벨 불필요
- 대규모 데이터셋: 언어 패턴 학습
- 문맥 이해: 'bank' (강둑/은행)
- 다음 문장 예측: 문장 관계
- MLM -> 강력한 표현

---

## ✍️ 자기회귀 생성

- 인과/디코더 모델: 다음 토큰 예측
- 입력: '고양이가 앉았다' -> '매트'를 예측
- 인과적 마스크: 미래 토큰 볼 수 없음
- 생성: 출력 분포에서 샘플링
- 온도: p'_i = p_i^(1/T) (무작위성 제어)
- 빔 서치: 상위 k개 시퀀스 유지
- Top-k 샘플링: 상위 k개 토큰에서 샘플
- Nucleus 샘플링: 누적 확률 p
- 길이 페널티: 짧은 시퀀스 방지
- 교사 강제: 학습 중 정답 사용
- 추론: 토큰별 자기회귀

### 코드 예제

#### Temperature Sampling

*온도 스케일링은 샘플링의 무작위성을 제어합니다: 낮을수록 결정적, 높을수록 랜덤합니다.*

```python
import torch
import torch.nn.functional as F

# 온도는 생성의 무작위성을 제어
logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -0.5])
print(f"로짓: {logits.tolist()}")

temperatures = [0.5, 1.0, 2.0]

print("\n다른 온도에서의 확률:")
for temp in temperatures:
    # 온도 적용: logits / T
    scaled_logits = logits / temp
    probs = F.softmax(scaled_logits, dim=-1)
    print(f"  T={temp}: {[f'{p:.3f}' for p in probs.tolist()]}")

print("\n낮은 T -> 더 확신 (뾰족함)")
print("높은 T -> 더 균일 (다양함)")

# 샘플링 시연
print("\n각 온도에서 10개 토큰 샘플링:")
for temp in temperatures:
    scaled_logits = logits / temp
    probs = F.softmax(scaled_logits, dim=-1)
    samples = torch.multinomial(probs, num_samples=10, replacement=True)
    print(f"  T={temp}: {samples.tolist()}")
```

**출력:**

```
로짓: [2.0, 1.0, 0.5, 0.10000000149011612, -0.5]

다른 온도에서의 확률:
  T=0.5: ['0.824', '0.111', '0.041', '0.018', '0.006']
  T=1.0: ['0.549', '0.202', '0.122', '0.082', '0.045']
  T=2.0: ['0.363', '0.220', '0.172', '0.141', '0.104']

낮은 T -> 더 확신 (뾰족함)
높은 T -> 더 균일 (다양함)

각 온도에서 10개 토큰 샘플링:
  T=0.5: [0, 0, 2, 0, 0, 0, 0, 1, 0, 1]
  T=1.0: [0, 4, 0, 3, 1, 3, 2, 1, 0, 0]
  T=2.0: [2, 0, 1, 3, 1, 4, 3, 0, 0, 3]
```

---

## 📈 스케일 & 창발

- 스케일링 법칙: 성능 ∝ 컴퓨트^α
- 파라미터: 100M -> 1B -> 100B -> 1T+
- 창발 능력: 스케일에서 나타남
- 인컨텍스트 학습: 학습 없이 few-shot
- 사고의 연쇄: 단계별 추론
- Grokking: 갑작스러운 일반화
- 컴퓨트 최적: Chinchilla 스케일링
- 데이터 스케일링: 더 많은 데이터 = 더 좋은 모델
- 모델 병렬: GPU에 분산
- 기울기 체크포인팅: 메모리 vs 컴퓨트
- 스케일이 모든 것을 바꾼다!

---

## 🔧 최적화 기법

- 기울기 누적: 큰 배치 시뮬레이션
- 혼합 정밀도: FP16 순전파, FP32 역전파
- 동적 손실 스케일링: 언더플로 방지
- 기울기 클리핑: ||g|| > 임계값
- 학습률 스케줄: 코사인, 다항식
- 가중치 감쇠: 파라미터별로 다르게
- 레이어별 학습률: 레이어마다 다르게
- 드롭아웃: 과적합 방지
- 라벨 스무딩: 타겟 분포 완화
- 데이터 증강: 다양성 증가
- 모든 기법이 스케일에서 중요!

---

## 🚀 현대적 개선

- RMSNorm: LayerNorm보다 간단
- SwiGLU: 트랜스포머에 더 좋은 활성화
- 회전 위치 임베딩: 더 나은 위치
- Flash Attention: 메모리 효율적 어텐션
- 기울기 체크포인팅: 메모리 절약
- 전문가 혼합: 조건부 계산
- 희소 어텐션: O(n²) 복잡도 감소
- 선형 어텐션: 어텐션 근사
- 그룹 쿼리 어텐션: 더 적은 KV 헤드
- 멀티 쿼리 어텐션: 공유 K,V
- 아키텍처의 지속적 혁신!

---
