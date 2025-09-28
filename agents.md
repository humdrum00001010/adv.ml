```adv ml/description.md
# 고급기계학습 팀플

## 1) 모델 구조

- 경량, 원클래스, timeseries 고려

### 입력·전처리

- 기본: 입술 랜드마크 시퀀스 $x \in \mathbb{R}^{T \times K \times 2}$
- 예: T=48∼64, K=20∼30 (lip부분 랜드마크 keypoints), 25–30fps, xy 좌표로 2차원
- (스무딩: EMA/칼만 살짝, 과도 스무딩 금지)

### 백본 인코더(공유) → LF/HF 두 개의 VAE 헤드

- 시간 인코더(랜드마크용, TCN 권장)
  - 1D TCN 3–4층, dilation [1,2,4,8], 채널 64→128
  - 출력: 프레임별 임베딩 $h_t \in \mathbb{R}^{128}$

- LF-VAE 헤드(clip-level 잠재)
  - Temporal pooling(mean+attention) → $g \in \mathbb{R}^{128}$
  - 선형층 → $\mu_{LF}, \log \sigma_{LF} \in \mathbb{R}^{32}$ (작은 차원)
  - LF-Decoder(작은 TCN or GRU)로 저역(느린 궤적) 복원 $\hat{x}_{LF}$

- HF-VAE 헤드(frame-level 잠재)
  - 각 $h_t$ → 선형층 → $\mu_t, \log \sigma_t \in \mathbb{R}^{64}$ (프레임별 잠재)
  - HF-Decoder(작은 TCN)로 고역 잔차/세부 복원 $\hat{r}_{HF}$

- 합성 출력(해석 쉬움)
  - 최종 복원: $\hat{x} = \hat{x}_{LF} + \hat{r}_{HF}$
  - 점수 계산은 $\hat{x}_{LF}$ 와 $\hat{r}_{HF}$ 의 오차를 분리해서 사용($E_{smooth}$, $E_{detail}$)

- 파라미터 수: 랜드마크 베이스 약 1.2–1.8M → 코랩 무료버전도 가

## 2) 로스 함수 설계 (저·고주파 역할 분리 + 시간 동역학 + 분해 해석성)

### 공통: VAE 기본

- 재파라미터화 $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- KL 항: $KL(q(z|x) \parallel \mathcal{N}(0, I))$

#### (A) LF(VAE) — 저주파만 잘 그리게

1. 저역 타깃 재구성 손실
   - 입력을 LPF(저역통과) 해서 타깃 $\tilde{x}_{LP}$ 생성
   - $L_{rec}^{LF} = \|\hat{x}_{LF} - \tilde{x}_{LP}\|_2^2$

2. 강한 β-KL (용량 축소)
   - $L_{KL}^{LF} = \beta_{LF} KL(q(z_{LF}|x), \mathcal{N}(0, I))$, $\beta_{LF} \approx 4 \sim 8$
   - → 고주파 정보는 병목을 못 통과

3. (선택) 시간 다운샘플/부드러움 유도
   - 디코더 입력 시간축 stride=2 설정 또는 $\|\nabla_t \hat{x}_{LF}\|_2^2$ 가벼운 가중

결과: LF는 개구/폐구, 모음 전이, 턱의 느린 회전 등만 복원

#### (B) HF(VAE) — 고주파/미세 동작을 정확히

1. 잔차/고역 타깃 재구성
   - 타깃 $r_{HF} = x - \tilde{x}_{LP}$ 또는 HPF(x)
   - $L_{rec}^{HF} = \|\hat{r}_{HF} - r_{HF}\|_2^2$

2. STFT 가중 손실(고주파 강조)
   - $L_{stft} = \sum_f w(f) \|STFT(\hat{r}_{HF}) - STFT(r_{HF})\|_2^2$
   - 여기서 상위 1/3 대역 $w(f)\uparrow$ (예 2–3배)

3. 시간 미분 손실(동역학 보존)
   - $L_{\nabla} = \lambda_1 \|\nabla_t \hat{x} - \nabla_t x\|_1 + \lambda_2 \|\nabla_t^2 \hat{x} - \nabla_t^2 x\|_1 + \lambda_3 \|\nabla_t^3 \hat{x} - \nabla_t^3 x\|_1$
   - jerk($\nabla_t^3$) 항은 자음 경계/급변에 민감

4. 약한 β-KL (정보 보존)
   - $L_{KL}^{HF} = \beta_{HF} KL(\cdot)$, $\beta_{HF} \approx 0.5 \sim 1.0$

5. (선택) 마스크드 시간 인페인팅
   - 랜덤 연속 프레임 마스크 후 복원 → 과적합/쇼트컷 방지
   - $L_{mask}$ (마스크 위치만 계산)

결과: HF는 립 코너 잔진동, 치아/혀 순간 하이라이트, 자음 폐쇄·폭발 등 미세 패턴을 복원

#### (C) LF/HF 분리와 일관성을 위한 보조 항

1. 합성 일치(전체 복원 체크)
   - $\hat{x} = \hat{x}_{LF} + \hat{r}_{HF}$ 가 원본과 맞는지:
   - $L_{full} = \|\hat{x} - x\|_1$ (작은 가중)
   - → 해석 가능한 분해 유지

2. 잠재 상관 억제(분리 강화)
   - LF 잠재를 시간으로 늘리고(↑) HF 잠재와 상호상관 최소화:
   - $L_{decor} = \|Corr(z_{LF}^\uparrow, z_{HF})\|_F^2$
   - 또는 HSIC/BarlowTwins식 크로스-코릴레이션 제약

3. 지터-일관성(실데이터 안정화)
   - 실(라이브) 클립에 ±1프레임 시간 워핑/프레임 드롭 5–10% 적용한 $x'$에 대해
   - HF 복원 일관성: $L_{jitter} = \|\hat{r}_{HF}(x') - \hat{r}_{HF}(x)\|_1$ (소가중)
   - → 실제 데이터에서는 ΔE가 작도록 학습(스푸프는 학습X → ΔE가 커지기 쉬움)

#### (D) 총손실 (권장 가중 예시)

$L = 1.0 \cdot L_{rec}^{LF} \underbrace{} + \beta_{LF}=6 \cdot L_{KL}^{LF} \underbrace{} + 0.5 \cdot L_{rec}^{HF} \underbrace{} + 0.5 \cdot L_{stft} \underbrace{} + 0.3 \cdot L_{\nabla} \underbrace{} + \beta_{HF}=0.8 \cdot L_{KL}^{HF} \underbrace{} + 0.2 \cdot L_{full} \underbrace{} + 0.1 \cdot L_{decor} \underbrace{} + 0.1 \cdot L_{jitter}$

- 계수는 개발셋에서 조정

---

## 3) 학습·추론 프로토콜 (요약)

- 학습 데이터: 실제(라이브)만 사용(원-클래스)
- 증강: 시간 지터(±1f), 프레임 드롭(5–10%), 속도 0.95–1.05×, 약한 공간 노이즈
- 스케줄: AdamW(lr 1e-3), cosine decay, batch 16–32, KL anneal(0→target, 10–20 epoch)

- 추론(평가):
  - $E_{smooth} = \|x - \hat{x}_{LF}\|$, $E_{detail} = \|r_{HF} - \hat{r}_{HF}\|$
  - $DRS = E_{smooth} / (E_{detail} + \epsilon)$
  - ΔE: 미세 지터 입력 후 HF 오차 증가량
  - C: 동역학 복잡도(샘플 엔트로피/스펙트럴 플랫니스)
  - 최종 점수: $S = \alpha \cdot DRS - \beta \cdot \Delta E + \gamma \cdot C$ → 임계값 τ로 Live/Spoof

## 참고논문

- [arXiv.org Generalized Face Liveness Detection via De-fake Face Generat…](https://arxiv.org/abs/2401.09006)
- [https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Khalid_OC-FakeDect_Classifying_Deepfakes_Using_One-Class_Variational_Autoencoder_CVPRW_2020_paper.pdf](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Khalid_OC-FakeDect_Classifying_Deepfakes_Using_One-Class_Variational_Autoencoder_CVPRW_2020_paper.pdf)

## 데이터셋

- [GRID Corpus Dataset (For training LipNet)](https://www.kaggle.com/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet)
- [Lip Reading Sentences 2 (LRS2) dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
- [AI-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=&topMenu=&srchOptnCnd=OPTNCND001&searchKeyword=%EB%A6%BD%EB%A6%AC%EB%94%A9&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=538)

---

## OC-FakeDect 이상치(Anomaly) 점수 요약

- 문제정의(원-클래스): 실사 얼굴만을 “정상”으로 학습하고, 나머지(딥페이크)를 “이상치”로 검출.
- 모델: One-Class VAE(OC-VAE)
  - 손실: 재구성 MSE + KL(q(z|x) || N(0, I))로 잠재 공간 정규화(재파라미터화 사용).
- 이상치 점수 정의(두 가지 변형)
  - OC-FakeDect-1: 입력 X와 재구성 X′의 RMSE(픽셀 공간).
  - OC-FakeDect-2: 인코더 평균 μ(X)와 “디코더 출력 X′를 다시 인코딩한” μ(X′) 간 RMSE(잠재 공간). 디코더 뒤 추가 인코더 블록을 사용.
- 임계값 설정(원-클래스 통계적 임계값):
  - 검증용 “실제(real)” 스코어 분포에서 IQR 기반으로 80% 분위 T80을 임계값으로 고정.
  - 점수 > T80 → 비정상(딥페이크)로 판정.
- 데이터/학습 세부(요약):
  - FaceForensics++에서 MTCNN으로 얼굴 검출·정렬, 100×100 리사이즈.
  - 증강: 수평/수직 플립, 밝기/휴/채도(±0.05) 변화, 정규화(mean=0.5, std=0.5).
  - Adam(lr=1e-3), batch=128, 약 300 epoch, best-val 기준 선택.
- 성능 요약/해석:
  - OC-AE 대비 전반적 우수. 일부 도메인(예: DF, NT)에서 잠재-RMSE(OC-FakeDect-2)가 근소하게 더 낫고, NT에서는 딥페이크를 높은 정확도로 검출.
- 우리 파이프라인에의 적용 포인트:
  - 현재 LF/HF 분해 기반 점수(E_smooth, E_detail)에 “픽셀/잠재 재구성 오차”를 추가하여 앙상블 가능.
  - 도메인별 개발셋에서 T(예: T80) 산출 → 운영 시 단일 임계값으로 경량 추론.
  - ΔE(지터 민감도)와 결합해 “고주파 불안정성”을 함께 보는 이중 기준이 실전 강건성 향상에 도움.

### 추가 참고자료

- CVF HTML: https://openaccess.thecvf.com/content_CVPRW_2020/html/w39/Khalid_OC-FakeDect_Classifying_Deepfakes_Using_One-Class_Variational_Autoencoder_CVPRW_2020_paper.html
- FaceForensics++(데이터셋) 개요: https://arxiv.org/abs/1901.08971
- VAE 원논문(Kingma & Welling): https://arxiv.org/abs/1312.6114
- One-Class SVM(Schölkopf et al., 2001): https://www.jmlr.org/papers/v2/scholkopf01a.html
- Grad-CAM(Selvaraju et al., 2016): https://arxiv.org/abs/1610.02391
- DASH-Lab paper list(OC-FakeDect 포함): https://github.com/DASH-Lab/deepfakeResearch
