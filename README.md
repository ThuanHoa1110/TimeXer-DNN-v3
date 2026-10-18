# 📌 BÁO CÁO CẬP NHẬT MÔ HÌNH: THAY ENCODER BACKBONE BẰNG MẠNG DNN-V3
> **Học phần:** Chuyên đề Trí tuệ nhân tạo và Khoa học dữ liệu  
> **Nhiệm vụ 1:** Code thay encoder (hoặc backbone) bằng DNN-v3. Submit GitHub kèm README chỉ chỗ đã update.  
> **Mã nguồn bài báo:** [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables (NeurIPS 2024)](https://github.com/thuml/TimeXer)

---

## 📑 Bảng tổng hợp các file và vị trí đã cập nhật (Update Highlights)

Dưới đây là danh sách chi tiết các tệp mã nguồn đã được tạo mới và chỉnh sửa theo đúng yêu cầu số 1 của giảng viên:

| STT | Tên File | Loại cập nhật | Vị trí / Chi tiết cập nhật | Mô tả mục đích |
|---|---|---|---|---|
| **1** | [`models/TimeXer_DnnV3.py`](./models/TimeXer_DnnV3.py) | **Tạo mới hoàn toàn** | Toàn bộ file | Triển khai phiên bản mô hình TimeXer độc lập với Encoder Backbone là mạng nơ-ron sâu `DNN-v3` (`DNN_v3_Encoder` & `DNN_v3_Layer`). |
| **2** | [`models/TimeXer.py`](./models/TimeXer.py) | **Chỉnh sửa** | Dòng 114 - 180, 200 - 240 | Thêm 2 class `DNN_v3_Layer`, `DNN_v3_Encoder` và cập nhật hàm `__init__` của `class Model` hỗ trợ cờ chọn `encoder_type='dnn_v3'` làm mặc định. |
| **3** | [`exp/exp_basic.py`](./exp/exp_basic.py) | **Chỉnh sửa** | Dòng 5 - 7, Dòng 38 - 40 | Import `TimeXer_DnnV3` và đăng ký 2 tên mô hình mới `'TimeXer_DnnV3'` và `'Dnn_v3'` vào từ điển `self.model_dict`. |
| **4** | [`run.py`](./run.py) | **Chỉnh sửa** | Dòng 26 - 28, Dòng 86 - 88 | Bổ sung đối số dòng lệnh `--encoder_type` (mặc định `'dnn_v3'`) và cập nhật danh sách mô hình hỗ trợ trong `--model`. |
| **5** | [`scripts/forecast_exogenous/EPF/TimeXer_DnnV3.sh`](./scripts/forecast_exogenous/EPF/TimeXer_DnnV3.sh) | **Tạo mới** | Toàn bộ file | Bash script cấu hình chạy huấn luyện và đánh giá mô hình `TimeXer_DnnV3` trên tập dữ liệu EPF (NP). |
| **6** | [`run_test_dnn_v3.py`](./run_test_dnn_v3.py) | **Tạo mới** | Toàn bộ file | Unit test script kiểm tra tính toàn vẹn của tensor flow, shape đầu vào/đầu ra cho cả nhiệm vụ đơn biến (MS) và đa biến (M). |
| **7** | [`README.md`](./README.md) | **Cập nhật** | Phần đầu README | Tài liệu báo cáo chính thức chỉ rõ toàn bộ các vị trí đã chỉnh sửa, sơ đồ phân tích kiến trúc mạng DNN-v3 và hướng dẫn chạy thực nghiệm theo yêu cầu của giảng viên. |
| **8** | [`.gitignore`](./.gitignore) | **Tạo mới** | Toàn bộ file | Cấu hình loại trừ các tệp rác, thư mục cache Python (`__pycache__/`, `*.pyc`), file checkpoint tạm để giữ repository sạch sẽ, chuẩn chỉ khi submit lên GitHub. |

---

## 🧠 Phân tích kiến trúc mạng DNN-v3 thay thế Transformer Encoder

Trong mô hình TimeXer nguyên bản:
- Encoder sử dụng các khối Transformer Layer gồm **Self-Attention** (tính toán phụ thuộc thời gian giữa các temporal patches của biến nội sinh) và **Cross-Attention** (kết nối token toàn cục với các biến ngoại sinh). Cơ chế Attention này đòi hỏi chi phí tính toán và bộ nhớ bậc 2 $O(N^2)$.

Trong kiến trúc **DNN-v3** mới thay thế:
Toàn bộ khối Transformer Attention được thay thế bằng mạng nơ-ron sâu đa tầng **3 giai đoạn (3-Stage Deep Neural Network)**:

```
[Input: Endogenous Patches + Global Token x, Exogenous Tokens cross]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Giai đoạn 1 (Stage 1): Intra-patch Temporal Feature MLP         │
│  - Linear(d_model -> d_ff) + Activation(GELU) + Dropout        │
│  - Linear(d_ff -> d_model) + Dropout                            │
│  - Residual Connection & LayerNorm: x = LayerNorm(x + res1)     │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Giai đoạn 2 (Stage 2): Cross-Variate Exogenous Fusion MLP       │
│  (Thay thế hoàn toàn cơ chế Cross-Attention)                    │
│  - Global Pooling & Exogenous Projection: cross -> [B, 1, D]    │
│  - Tách token toàn cục nội sinh: x_glb = x[:, -1:, :]           │
│  - Deep Interaction MLP: Linear(2*D -> d_ff) -> GELU -> Dropout │
│                          Linear(d_ff -> D)                      │
│  - Residual & LayerNorm: x_glb = LayerNorm(x_glb + fused)       │
│  - Tái ghép chuỗi: x = [patches, x_glb]                         │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Giai đoạn 3 (Stage 3): Deep Feed-Forward Expansion MLP          │
│  - Linear(d_model -> d_ff) + Activation(GELU) + Dropout        │
│  - Linear(d_ff -> d_model) + Dropout                            │
│  - Residual Connection & Final LayerNorm: out = LayerNorm(x + y)│
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
        [Output: Hidden Representations for FlattenHead]
```

### Ưu điểm vượt trội của DNN-v3:
1. **Bảo tồn khả năng học biến ngoại sinh:** Vẫn duy trì cơ chế cốt lõi của TimeXer trong việc dung nạp thông tin từ các biến ngoại cảnh (exogenous variables) thông qua mạng kết nối sâu ở Giai đoạn 2.
2. **Tốc độ suy luận và tối ưu hóa cao hơn:** Không phụ thuộc ma trận trọng số Attention $O(N^2)$, DNN-v3 có chi phí tính toán tuyến tính $O(N \cdot D)$, giảm đáng kể thời gian huấn luyện và lượng bộ nhớ GPU tiêu thụ.
3. **Độ ổn định cao:** Được trang bị chuẩn hóa `LayerNorm` ở từng giai đoạn cùng các liên kết tắt `Residual Connections` giúp gradient lan truyền sâu mà không bị suy hao (vanishing gradient).

---

## 💻 Hướng dẫn chạy thử nghiệm mô hình DNN-v3

### Cách 1: Chạy script kiểm tra kiến trúc (Unit Test)
```bash
python run_test_dnn_v3.py
```

### Cách 2: Chạy trực tiếp qua lệnh dòng lệnh `run.py`
```bash
python run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path NP.csv \
  --model_id NP_168_24_DnnV3 \
  --model TimeXer_DnnV3 \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des 'TimeXer_DnnV3' \
  --patch_len 24 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --itr 1
```

### Cách 3: Chạy bằng Shell Script có sẵn
```bash
bash ./scripts/forecast_exogenous/EPF/TimeXer_DnnV3.sh
```

---

## 📤 Hướng dẫn sinh viên Submit GitHub
1. Khởi tạo repository hoặc link tới repo cá nhân của bạn:
   ```bash
   git add .
   git commit -m "Update Yeu Cau 1: Thay the encoder backbone bang DNN-v3 kem cap nhat README"
   git branch -M main
   git remote add origin <URL_REPO_GITHUB_CUA_BAN>
   git push -u origin main
   ```
2. Copy link GitHub của bạn để nộp vào biểu mẫu / file PDF tổng kết theo yêu cầu của thầy.

---

# TimeXer Original Documentation

This repo is the official implementation for the paper: [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2402.19072).

## Introduction
This paper focuses on forecasting with exogenous variables which is a practical forecasting paradigm applied extensively in real scenarios. TimeXer empower the canonical Transformer with the ability to reconcile endogenous and exogenous information without any architectural modifications and achieves consistent state-of-the-art performance on twelve real-world forecasting benchmarks.

<p align="center">
<img src=".\figures\Introduction.png" width = "800" height = "" alt="" align=center />
</p>

## Overall Architecture
TimeXer employs patch-level and variate-level representations respectively for endogenous and exogenous variables, with an endogenous global token as a bridge in-between. With this design, TimeXer can jointly capture intra-endogenous temporal dependencies and exogenous-to-endogenous correlations.

<p align="center">
<img src=".\figures\TimeXer.png" width = "800" height = "" alt="" align=center />
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
<img src=".\figures\Result_EPF.png" width = "800" height = "" alt="" align=center />
</p>

### Multivariate Forecasting

<p align="center">
<img src=".\figures\Result_Multivariate.png" width = "800" height = "" alt="" align=center />
</p>

## Experiments on Large-scale Meteorology Dataset
In this paper, we build a large-scale weather dataset for forecasting with exogenous variables, where the endogenous series is the hourly temperature of 3,850 stations worldwide obtained from the National Centers for Environmental Information (NCEI), and the exogenous variables are meteorological indicators of its adjacent area from the ERA5 dataset. You can obtain this meteorology dataset from [[Google Drive]](https://drive.google.com/file/d/1EuEedepUV2A_cia1plAHwA6fJXNio47i/view?usp=drive_link).

<p align="center">
<img src=".\figures\ERA5.png" width = "800" alt="" align=center />
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

Stationary (https://github.com/thuml/Nonstationary_Transformers)

Time-Series-Library (https://github.com/thuml/Time-Series-Library)

## Concat

If you have any questions or want to use the code, please contact wangyuxu22@mails.tsinghua.edu.cn
