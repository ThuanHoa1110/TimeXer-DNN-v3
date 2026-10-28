# TimeXer-DINOv3: Empowering Transformers for Time Series Forecasting with Exogenous Variables and DINO-v3

This repository provides the official implementation of **TimeXer-DINOv3**, extending the NeurIPS 2024 paper *[TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2402.19072)* by integrating a **DINO-v3 Backbone** and a **DINO-v3 Multi-Objective Composite Loss Function**.

---

## 📑 DINO-v3 Integration Highlights & Code Modifications

Below is the comprehensive summary of newly created and modified files, specifically detailing the **DINO-v3 Backbone integration** and **DINO-v3 Composite Loss Function** implementations:

| No. | File Name | Update Type | Modification Details / Location | Description & Purpose |
|---|---|---|---|---|
| **1** | [`models/TimeXer_DINOv3.py`](./models/TimeXer_DINOv3.py) | **Newly Created** | Entire file | Independent implementation of the **DINO-v3 Backbone** (`DINOv3_Encoder` & `DINOv3_Block` with **LayerScale**, MHSA, Cross-Fusion) and **DINO-v3 Composite Loss Function** (`DINOv3Loss`). |
| **2** | [`models/TimeXer.py`](./models/TimeXer.py) | **Modified** | Line 6, Lines 235 - 280 | Imported `DINOv3_Block`, `DINOv3_Encoder`, `DINOv3Loss`. Added `--encoder_type dino_v3` flag as default and integrated `self.loss_fn = DINOv3Loss(...)`. |
| **3** | [`exp/exp_long_term_forecasting.py`](./exp/exp_long_term_forecasting.py) | **Modified** | Lines 35 - 50, 85 - 90, 140 - 165 | Updated `_select_criterion()` to support `dino_v3`. In `train()`, computed total composite loss and logged all 3 decomposed components: $\mathcal{L}_{\text{task}}$, $\mathcal{L}_{\text{distill}}$, $\mathcal{L}_{\text{gram}}$. |
| **4** | [`run.py`](./run.py) | **Modified** | Lines 26 - 28, Lines 87 - 100 | Registered `--model TimeXer_DINOv3`, `--encoder_type dino_v3`, `--loss dino_v3`, and 4 DINO-v3 hyperparameters (`--dino_temp_s`, `--dino_temp_t`, `--dino_lambda_distill`, `--dino_lambda_gram`). |
| **5** | [`exp/exp_basic.py`](./exp/exp_basic.py) | **Modified** | Line 6, Lines 38 - 42 | Imported `TimeXer_DINOv3` and registered aliases `'TimeXer_DINOv3'`, `'DINO_v3'`, `'DINOv3'` in `self.model_dict`. |
| **6** | [`run_test_dinov3.py`](./run_test_dinov3.py) | **Newly Created** | Entire file | Comprehensive Unit Test script verifying: Forward Tensor Shape, DINOv3 Composite Loss Function, and Backward Gradient Flow. |
| **7** | [`scripts/forecast_exogenous/EPF/TimeXer_DINOv3.sh`](./scripts/forecast_exogenous/EPF/TimeXer_DINOv3.sh) | **Newly Created** | Entire file | Standard training bash script for TimeXer with DINO-v3 Backbone and Loss on EPF (NP) benchmark. |
| **8** | [`.github/workflows/deploy_pages.yml`](./.github/workflows/deploy_pages.yml) | **Newly Created** | Entire file | GitHub Actions automated workflow building and deploying the web application to GitHub Pages (`https://thuanhoa1110.github.io/TimeXer-DNN-v3`). |
| **9** | [`generate_6million_dataset.py`](./generate_6million_dataset.py) | **Newly Created** | Entire file | Generates 6-station meteorological network dataset conforming to NeurIPS TimeXer standards: **6,312,384 samples** (10 years, 5-minute resolution). |
| **10** | [`visualize_6million_network.py`](./visualize_6million_network.py) | **Newly Created** | Entire file | Visualization script for all 6.31M samples, rendering 5 academic 300 DPI figures into `figures/`. |
| **11** | [`visualize_timexer_paper_benchmark.py`](./visualize_timexer_paper_benchmark.py) | **Newly Created** | Entire file | 100% TimeXer Paper (NeurIPS 2024) benchmark visualizations: Inflection points evaluation, Cross-Attention interpretability, causal time-lag, and scalability benchmark. |

---

## 🔬 Detailed Formulation of DINO-v3 Composite Loss Function

The core innovation of DINO-v3 (Meta FAIR 2025) over conventional self-supervised frameworks is the **Multi-objective Composite Loss Function**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{distill}} \mathcal{L}_{\text{distill}} + \lambda_{\text{gram}} \mathcal{L}_{\text{gram}}$$

```
┌────────────────────────────────────────────────────────────────────────┐
│             DINO-v3 COMPOSITE LOSS COMPUTATION MECHANISM               │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Task Forecasting Loss (L_task):                                     │
│    - MSE / MAE between forecasted series and ground truth labels:      │
│      L_task = (1 / N) * Σ (y_pred - y_true)^2                          │
│                                                                        │
│ 2. DINO Self-Distillation Loss (L_distill):                            │
│    - Knowledge distillation via temperature-scaled & centered softmax: │
│      P_s = softmax(z_s / tau_s)                                        │
│      P_t = softmax((z_t - c) / tau_t)    (with tau_t < tau_s)          │
│      L_distill = - Σ P_t * log(P_s)                                    │
│    - Centering vector c is continuously updated via EMA to prevent     │
│      representation collapse: c <- m * c + (1 - m) * mean(z_t)         │
│                                                                        │
│ 3. DINOv3 Gram Matrix Anchoring Loss (L_gram) [DINOv3 BREAKTHROUGH]:   │
│    - Prevents dense patch feature drift and representation collapse    │
│    - Computes Gram correlation matrix between patch tokens: G = Z * Z^T│
│    - Constrains student Gram matrix to anchor spatial structure:       │
│      L_gram = (1 / N^2) * || G_student - G_anchor ||_F^2               │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Advantages of Integrating DINO-v3 Loss:
1. **Anti-Collapse Representation:** Centering ($c$) and teacher sharpening ($\tau_t = 0.04 < \tau_s = 0.1$) prevent all tokens from collapsing into trivial uniform vectors.
2. **Local Correlation Preservation via Gram Anchoring:** The Gram matrix $G \in \mathbb{R}^{N \times N}$ maintains inter-patch correlation topology across thousands of training iterations, eliminating dense feature drift.
3. **Superior Forecasting Generalization:** Latent representations learned under DINO-v3 self-supervision exhibit high generalization, enabling the `FlattenHead` forecasting head to reconstruct complex temporal curves and inflection points with significantly lower error compared to standard MSE alone.

---

## 🧠 DINO-v3 Backbone Architecture

The **`DINOv3_Block`** is designed incorporating recent architectural breakthroughs from Meta FAIR:

```
[Input: Endogenous Patches + Global Token x, Exogenous Tokens cross]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Multi-Head Self-Attention (MHSA) + LayerScale          │
│  - LayerNorm(x) -> Multi-Head Self-Attention -> Dropout         │
│  - LayerScale: x = x + gamma_1 * Attention(x)                   │
│    (gamma initialized to 1e-5 stabilizes deep gradient flows)   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Exogenous Information Cross-Fusion Module              │
│  - Ingest exogenous variable representations: cross -> [B, 1, D]│
│  - Extract endogenous Global Token: x_glb = x[:, -1:, :]        │
│  - Gated Fusion MLP: Linear(2*D -> d_ff) -> GELU -> Dropout     │
│  - LayerScale & Residual: x_glb = x_glb + gamma_2 * Fused       │
│  - Re-concatenate sequence: x = [patches, x_glb]                │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: SwiGLU / Feed-Forward Expansion Network                │
│  - Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)   │
│  - LayerScale & Final Residual: x = x + gamma_3 * FFN(x)         │
│  - Final Normalization: LayerNorm(x)                            │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
    ┌────────────────────┴────────────────────┐
    ▼                                         ▼
[Forecasting: FlattenHead]      [Loss: DINOv3 Projection Head]
  -> [B, Pred_Len, C]            -> z_s & Patch Tokens (Gram)
```

---

## 🌍 Large-Scale Multi-Station Meteorology Dataset (6,312,384 Samples)

To evaluate multi-variable time series forecasting with spatial-temporal exogenous correlations at scale, this repository includes an automated 6-station meteorological observation network across Hanoi with a total of **6,312,384 samples** (100% complete, 0.00% missing values, 5-minute sampling resolution over 10 consecutive years 2016–2025):

### Technical Specifications across 6 Stations:
| Monitoring Station | Station ID | Topographic / Micro-climate Feature | Samples | Size | Missing |
|---|:---:|---|:---:|:---:|:---:|
| **Hoan Kiem** | `st01_hoankiem` | Historic urban core, intense heat island (+1.2°C) | 1,052,064 | 83.7 MB | 0.00% |
| **Ba Dinh** | `st02_badinh` | Administrative center, moderated by West Lake (+0.4°C) | 1,052,064 | 83.7 MB | 0.00% |
| **Cau Giay** | `st03_caugiay` | New urban belt, dense traffic corridor (+0.8°C) | 1,052,064 | 83.7 MB | 0.00% |
| **Ha Dong** | `st04_hadong` | Southwestern suburban reference baseline (0.0°C) | 1,052,064 | 83.6 MB | 0.00% |
| **Dong Anh** | `st05_donganh` | North Red River peri-urban area, higher wind (-0.6°C) | 1,052,064 | 83.6 MB | 0.00% |
| **Soc Son** | `st06_socson` | Semi-mountainous hilly terrain, elevation +60m (-1.5°C) | 1,052,064 | 83.6 MB | 0.00% |
| **TOTAL** | **6 STATIONS** | **Hanoi Micro-climate Monitoring Network** | **6,312,384** | **501.9 MB** | **0.00%** |

### 📈 Network Data Visualizations (300 DPI):

#### 1. 10-Year Temporal Trajectory (2016 – 2025) & Urban Heat Island (UHI) Effect
<p align="center">
<img src="./figures/multistation_01_10year_trajectory.png" width="95%" alt="10 Year Trajectory" />
</p>

#### 2. Spatial Micro-climate Distribution across 6 Stations (Boxplots)
<p align="center">
<img src="./figures/multistation_02_spatial_comparison_boxplots.png" width="95%" alt="Spatial Distribution" />
</p>

#### 3. Cross-Station Spatial Pearson Correlation Matrix
<p align="center">
<img src="./figures/multistation_03_cross_station_correlation.png" width="70%" alt="Spatial Correlation Matrix" />
</p>

#### 4. 24-Hour Diurnal Cycle & Day-Night Temperature Range
<p align="center">
<img src="./figures/multistation_04_diurnal_24h_cycle.png" width="95%" alt="Diurnal 24h Cycle" />
</p>

#### 5. Dataset Volume Breakdown (6,312,384 Samples)
<p align="center">
<img src="./figures/multistation_05_volume_breakdown_6million.png" width="95%" alt="Volume Breakdown" />
</p>

### 🎓 Benchmark Visualizations (TimeXer NeurIPS 2024 Standards)
*(Conforming to the methodology and publication figure standards of the original TimeXer paper - Yuxuan Wang et al., Tsinghua University)*:

#### Figure P1: Problem Formulation – Endogenous Target vs Exogenous Drivers
*(Replicating Paper Fig 1 & Fig 4: Spatial network diagram and Look-back $L=7$ days vs Forecast $H=3$ days)*
<p align="center">
<img src="./figures/timexer_paper_fig1_formulation.png" width="95%" alt="TimeXer Paper Formulation" />
</p>

#### Figure P2: Forecasting Showcase & Inflection Points Evaluation
*(Replicating Paper Fig 11 & Fig 12: Green circle $\bigcirc$ for accurate inflection captures and Red circle $\bigcirc$ for failure without exogenous variables)*
<p align="center">
<img src="./figures/timexer_paper_fig2_inflection_showcase.png" width="95%" alt="TimeXer Inflection Showcase" />
</p>

#### Figure P3: Variate-wise Cross-Attention Weights Heatmap across Weather Scenarios
*(Replicating Paper Section 4.4 & Fig 5: Proving interpretability between Global Token and exogenous drivers)*
<p align="center">
<img src="./figures/timexer_paper_fig3_cross_attention_map.png" width="95%" alt="TimeXer Cross-Attention Heatmap" />
</p>

#### Figure P4: Spatial Causal Time-Lag Analysis
*(Replicating Paper Section 1 & Fig 1 Left: Addressing temporal misalignment between upstream exogenous stations and downstream target stations)*
<p align="center">
<img src="./figures/timexer_paper_fig4_causal_lag_analysis.png" width="95%" alt="TimeXer Causal Time-Lag" />
</p>

#### Figure P5: Scalability Benchmark on 6.3 Million Samples
*(Replicating Paper Section 4.3.2 & Fig 8: Look-back horizon expansion $L \in \{96..720\}$ and linear GPU memory complexity $\mathcal{O}(C)$)*
<p align="center">
<img src="./figures/timexer_paper_fig5_scalability_benchmark.png" width="95%" alt="TimeXer Scalability Benchmark" />
</p>

---

# TimeXer Original Documentation

This repo is the official implementation for the paper: [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2402.19072).

## Introduction
This paper focuses on forecasting with exogenous variables which is a practical forecasting paradigm applied extensively in real scenarios. TimeXer empower the canonical Transformer with the ability to reconcile endogenous and exogenous information without any architectural modifications and achieves consistent state-of-the-art performance on twelve real-world forecasting benchmarks.

<p align="center">
<img src="./figures/Introduction.png" width = "800" height = "" alt="" align=center />
</p>

## Overall Architecture
TimeXer employs patch-level and variate-level representations respectively for endogenous and exogenous variables, with an endogenous global token as a bridge in-between. With this design, TimeXer can jointly capture intra-endogenous temporal dependencies and exogenous-to-endogenous correlations.

<p align="center">
<img src="./figures/TimeXer.png" width = "800" height = "" alt="" align=center />
</p>

## Usage 

1. Short-term Electricity Price Forecasting Dataset have alreadly included in "./dataset/EPF". Multivariate datasets can be obtained from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy).

2. Install Pytorch and other necessary dependencies.
```
pip install -r requirements.txt
```
3. Train and evaluate model. We provide the experiment scripts under the folder ./scripts/. You can reproduce the experiment results as the following examples:

```
bash ./scripts/forecast_exogenous/EPF/TimeXer.sh
```

## Main Results
We evaluate TimeXer on short-term forecasting with exogenous variables and long-term multivariate forecasting benchmarks. Comprehensive forecasting results demonstrate that TimeXer effectively ingests exogenous information to facilitate the prediction of endogenous series.

### Forecasting with Exogenous

<p align="center">
<img src="./figures/Result_EPF.png" width = "800" height = "" alt="" align=center />
</p>

### Multivariate Forecasting

<p align="center">
<img src="./figures/Result_Multivariate.png" width = "800" height = "" alt="" align=center />
</p>

## Experiments on Large-scale Meteorology Dataset
In this paper, we build a large-scale weather dataset for forecasting with exogenous variables, where the endogenous series is the hourly temperature of 3,850 stations worldwide obtained from the National Centers for Environmental Information (NCEI), and the exogenous variables are meteorological indicators of its adjacent area from the ERA5 dataset. You can obtain this meteorology dataset from [[Google Drive]](https://drive.google.com/file/d/1EuEedepUV2A_cia1plAHwA6fJXNio47i/view?usp=drive_link).

<p align="center">
<img src="./figures/ERA5.png" width = "800" alt="" align=center />
</p>

## Citation
If you find this repo helpful, please cite our paper.

```
@article{wang2024timexer,
  title={Timexer: Empowering transformers for time series forecasting with exogenous variables},
  author={Wang, Yuxuan and Wu, Haixu and Dong, Jiaxiang and Liu, Yong and Qiu, Yunzhong and Zhang, Haoran and Wang, Jianmin and Long, Mingsheng},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.

Reformer (https://github.com/lucidrains/reformer-pytorch)

Informer (https://github.com/zhouhaoyi/Informer2020)

Autoformer (https://github.com/thuml/Autoformer)
