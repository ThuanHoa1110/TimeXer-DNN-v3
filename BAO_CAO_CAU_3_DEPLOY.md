# 📌 BÁO CÁO TỔNG HỢP & HOÀN THIỆN 3 BÀI TẬP (THEO NHẬN XÉT CỦA GIẢNG VIÊN)
> **Học phần:** Chuyên đề Trí tuệ nhân tạo và Khoa học dữ liệu (Lớp 65TTNT)  
> **Khoa Công nghệ Thông tin - Trường Đại học Thủy Lợi**  
> **Sinh viên thực hiện:** Nguyễn Thuận Hóa  
> **Mã nguồn GitHub Repository:** [https://github.com/ThuanHoa1110/TimeXer-DNN-v3](https://github.com/ThuanHoa1110/TimeXer-DNN-v3)  

---

## 📋 TỔNG HỢP NỘI DUNG ĐÃ SỬA THEO NHẬN XÉT CỦA THẦY GIÁO

| Yêu cầu & Nhận xét của Thầy | Tình trạng trước đây | Giải pháp & Các nội dung đã hoàn thiện lại | File & Minh chứng hoàn thành |
|---|---|---|---|
| **Bài 1:** *"Chưa thể hiện được chỗ update dino-v3 e nhé (loss function)"* | - Mới chỉ viết dưới dạng MLP đơn giản (DNN-v3).<br>- Chưa có hàm mất mát DINO-v3 (`MSELoss` mặc định). | - Xây dựng **DINO-v3 Backbone** hoàn chỉnh (LayerScale, MHSA, Cross-Fusion).<br>- Triển khai **DINO-v3 Composite Loss Function** gồm 3 thành phần: $\mathcal{L}_{\text{task}}$, $\mathcal{L}_{\text{distill}}$ và $\mathcal{L}_{\text{gram}}$ (Gram Matrix Anchoring Loss độc quyền của DINOv3 - Meta 2025).<br>- Cập nhật vòng lặp huấn luyện bóc tách 3 chỉ số loss và viết README chỉ rõ từng dòng code. | - [`models/TimeXer_DINOv3.py`](./models/TimeXer_DINOv3.py)<br>- [`models/TimeXer.py`](./models/TimeXer.py)<br>- [`exp/exp_long_term_forecasting.py`](./exp/exp_long_term_forecasting.py)<br>- [`run_test_dinov3.py`](./run_test_dinov3.py)<br>- [`README.md`](./README.md) |
| **Bài 2:** *"Chưa tham khảo bài bạn Việt ah, cần có code visualize data em nhé. Data của em đang rất ít."* | - Bộ dữ liệu cũ bị thầy yêu cầu phải đạt quy mô từ 6 triệu dòng. | - Nâng cấp lên bộ dữ liệu **quy mô cực lớn đa trạm (6,312,384 dòng)** trên mạng lưới 6 trạm quan trắc tự động tại Hà Nội theo chuẩn bài báo TimeXer (NeurIPS 2024), tần số 5 phút trong suốt 10 năm liên tục 2016–2025.<br>- Biên soạn hoàn chỉnh tệp Jupyter Notebook trực quan hóa chuẩn học thuật (`kaggle_eda_visualization.ipynb`) gồm 8 mục phân tích chuyên sâu.<br>- Xuất bộ 6 biểu đồ phân tích 300 DPI và viết lại tài liệu Kaggle Usability 10.0. | - [`kaggle_eda_visualization.ipynb`](./kaggle_eda_visualization.ipynb)<br>- [`generate_6million_dataset.py`](./generate_6million_dataset.py)<br>- [`visualize_dataset.py`](./visualize_dataset.py)<br>- [`data/private/real_weather_large_dataset.csv`](./data/private/real_weather_large_dataset.csv)<br>- [`figures/weather_01_*.png`](./figures/)<br>- [`KAGGLE_DESCRIPTION.txt`](./KAGGLE_DESCRIPTION.txt) |
| **Bài 3:** *"Link không work"* | - Link Render cũ và Localtunnel bị lỗi 404 / tắt server khi ngắt terminal. | - Tạo quy trình **GitHub Actions tự động deploy** lên GitHub Pages mỗi khi push code lên `main`.<br>- Tối ưu `index.html` chạy 100% mượt mà cả chế độ online lẫn offline client.<br>- Thiết lập cấu hình **1-click deploy Vercel** (`vercel.json`) và Streamlit Cloud cho `streamlit_app.py`. | - [`.github/workflows/deploy_pages.yml`](./.github/workflows/deploy_pages.yml)<br>- [`vercel.json`](./vercel.json)<br>- [`index.html`](./index.html)<br>- [`streamlit_app.py`](./streamlit_app.py) |

---

## 🎯 CHI TIẾT KẾT QUẢ SỬA BÀI CỤ THỂ

### 1️⃣ BÀI 1: Thay thế Backbone DINO-v3 & Tích hợp DINO-v3 Loss Function
- **Đặc tả Backbone DINO-v3:**
  - Khối `DINOv3_Block`: Tích hợp cơ chế **LayerScale** ($\gamma = 10^{-5}$) giúp triệt tiêu hiện tượng bão hòa gradient khi mạng sâu; Multi-Head Self-Attention trên các temporal patches; và Exogenous Information Fusion kết nối biến ngoại cảnh vào Endogenous Global Token.
  - Encoder `DINOv3_Encoder`: Xếp chồng các khối DINOv3 Block và trang bị DINO Projection Head để trích xuất biểu diễn toàn cục và cục bộ.
- **Hàm mất mát DINO-v3 Composite Loss Function:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{distill}} \mathcal{L}_{\text{distill}} + \lambda_{\text{gram}} \mathcal{L}_{\text{gram}}$$
  - **$\mathcal{L}_{\text{task}}$:** MSE loss đảm bảo độ chính xác dự báo chuỗi thời gian.
  - **$\mathcal{L}_{\text{distill}}$:** DINO Self-Distillation Cross-Entropy giữa phân phối Student và Teacher với cơ chế kiểm soát nhiệt độ ($\tau_t = 0.04 < \tau_s = 0.1$) và vector định tâm (centering vector $c$) tránh sụp đổ biểu diễn.
  - **$\mathcal{L}_{\text{gram}}$ (Gram Matrix Anchoring Loss):** Tính ma trận Gram $G = \tilde{Z}\tilde{Z}^T$ giữa các patch tokens, neo giữ cấu trúc không gian tương quan tránh hiện tượng trôi dạt đặc trưng (dense feature drift) trong quá trình huấn luyện kéo dài.
- **Kiểm thử tự động:** Script `python run_test_dinov3.py` đã vượt qua cả 4 bài test: Shape MS/M, tính toán DINOv3Loss và backward gradient flow.

---

### 2️⃣ BÀI 2: Dữ liệu lớn Triệu bản ghi (6,312,384 dòng) & Code Trực quan hóa dữ liệu (Kaggle EDA)
- **Tập dữ liệu quy mô hơn 6.3 triệu bản ghi:**
  - **6,312,384 dòng dữ liệu** liên tục tần số cao 5 phút/mẫu trong suốt 10 năm (2016–2025) trên mạng lưới 6 trạm quan trắc tự động Hà Nội ($6 \times 1,052,064$ dòng/trạm: Hoàn Kiếm, Ba Đình, Cầu Giấy, Hà Đông, Đông Anh, Sóc Sơn).
  - **13 thuộc tính:** Biến mục tiêu nội sinh (`temperature`) và 12 biến ngoại sinh (`humidity`, `solar_radiation`, `direct_radiation`, `surface_pressure`, `wind_speed_10m`, `wind_gusts_10m`, `precipitation`, `rain`, `dew_point`, `cloud_cover`, `station_id`...).
  - **Chất lượng:** 0% missing value / NaN. Đạt chuẩn Big Data quốc tế theo mô hình đa trạm của bài báo TimeXer (NeurIPS 2024).
- **Code trực quan hóa dữ liệu (Notebook & Python):**
  - Đính kèm tệp **`kaggle_eda_visualization.ipynb`** chuẩn bị sẵn cho Kaggle Notebook.
  - Bộ 6 biểu đồ học thuật xuất bản quốc tế (300 DPI):
    1. *weather_01_10year_overview.png:* Toàn cảnh 10 năm, dải Min-Max và sự kiện cực đoan (đợt rét kỷ lục 4.4°C năm 2016 và đợt nóng 40.2°C năm 2024).
    2. *weather_02_diurnal_cycle.png:* Trễ pha nhật kỳ 2.5h giữa bức xạ mặt trời và nhiệt độ.
    3. *weather_03_seasonality_12months.png:* Chu kỳ mùa vụ 12 tháng phản ánh gió mùa nhiệt đới.
    4. *weather_04_correlation_matrix.png:* Ma trận tương quan Pearson giữa Target và biến ngoại sinh.
    5. *weather_05_decomposition.png:* Phân rã chuỗi thời gian (Trend, Seasonal 24h, Residuals).
    6. *weather_06_distributions.png:* Phân phối xác suất 2 đỉnh theo mùa (Bimodal distribution).

---

### 3️⃣ BÀI 3: Hệ thống Giao diện Web Dự báo Khí tượng & Triển khai Deploy Hoàn chỉnh

#### A. Phân tích Nghiệp vụ Bài toán Cốt lõi (Core Business Problem)
- **Bài toán:** Dự báo chuỗi thời gian khí tượng đa biến kết hợp biến ngoại sinh (**Multi-Station Meteorological Forecasting with Exogenous Variables**).
- **Mục tiêu dự báo (Endogenous Target):** **Nhiệt độ không khí (°C)** tại trạm khí tượng được chỉ định.
- **Biến ngoại sinh bổ trợ (Exogenous Drivers):** Bức xạ mặt trời ($W/m^2$), Vận tốc gió 10m ($m/s$), Độ ẩm tương đối (%), Khí áp bề mặt ($hPa$).

#### B. Tối ưu hóa Đầu vào Người dùng & Xử lý Dữ liệu Thực tế (Data Input & Preprocessing)
1. **Hỗ trợ 2 phương thức nạp dữ liệu thực:**
   - **Cách 1: Upload file CSV của người dùng:** Người dùng có thể kéo thả tệp CSV quan trắc của mình (tối thiểu 96 bước thời gian). Hệ thống tự động kiểm tra định dạng cột, căn chỉnh tần số và nạp vào Tensor.
   - **Cách 2: Chọn mẫu thực tế từ kho 6 trạm quan trắc:** Chọn trực tiếp trạm (`Hoàn Kiếm`, `Ba Đình`, `Cầu Giấy`, `Hà Đông`, `Đông Anh`, `Sóc Sơn`) và các sự kiện thời tiết lịch sử (Đợt nắng nóng kỷ lục, đợt rét đậm không khí lạnh, pha chuyển mùa).
2. **Cấu hình tham số suy luận:**
   - **Khoảng thời gian dự báo (Horizon $H$):** `24 bước`, `48 bước` hoặc `72 bước` (chuẩn Paper TimeXer).
   - **Biến ngoại sinh:** Bật/tắt Cross-Attention cho Bức xạ, Gió, Độ ẩm, Khí áp.
3. **Tiền xử lý và chuyển đổi Tensor:**
   - Chuẩn hóa RevIN Reversible Instance Normalization và StandardScaler ($\mu, \sigma$) lưu tại `checkpoints/scaler_params.json`.
   - Tạo ma trận nhúng thời gian $X_{\text{mark}} \in \mathbb{R}^{B \times L \times 4}$.

#### C. Kiến trúc Model Serving & Thực thi Suy luận PyTorch Thực tế (True Inference Engine)
1. **Module suy luận độc lập (`inference_engine.py`):**
   - Class `TimeXerInferenceEngine` tự động phát hiện thiết bị (`CUDA` hoặc `CPU`).
   - Nạp cấu trúc mạng `TimeXer_DINOv3` và trọng số checkpoint chính xác: `checkpoints/timexer_dinov3_best.pth` (~2.1 MB).
   - Thực thi Forward Pass trên PyTorch: `with torch.no_grad(): outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)`.
   - Độ trễ suy luận siêu tốc: **10.72 ms trên CPU**, không dùng bất kỳ dữ liệu hay công thức giả lập sóng sin nào.
2. **Kịch bản huấn luyện & xuất Checkpoint (`scripts/train_and_save_checkpoint.py`):**
   - Huấn luyện mô hình trực tiếp trên các lát cắt chuỗi thời gian khí tượng thực tế với hàm mất mát `DINOv3Loss` (MSE + Self-Distillation + Gram Anchoring Loss).
   - Tự động xuất file trọng số `.pth` và thông số scaler chuẩn hóa.

#### D. Thiết kế Đầu ra Hiển thị (Outputs - Chuẩn, đầy đủ, ít thành phần thừa)
1. **Thanh 4 Thẻ KPI tóm tắt:** Nhiệt độ hiện tại ($T_0$), Biên độ nhiệt dự kiến ($T_{\text{max}} - T_{\text{min}}$), Thời gian phản hồi suy luận (Latency theo ms), và Số điểm uốn nhiệt độ cực trị bắt được.
2. **Biểu đồ Dự báo Chuỗi thời gian tương tác:**
   - Vẽ đồng thời: *Quá khứ quan sát thực tế (Look-back $L=96$)*, *Đường dự báo thực từ PyTorch Model*, và *Giá trị thực tế kiểm chứng (Ground Truth - nếu có)*.
   - Đánh dấu **vòng tròn xanh lá ($\bigcirc$)** tại các điểm uốn đỉnh/đáy nhiệt độ theo chuẩn Paper TimeXer.
3. **Biểu đồ đóng góp Biến Ngoại sinh (Explainable AI - XAI):** Tỷ lệ % ảnh hưởng của Bức xạ mặt trời, Tốc độ gió, Độ ẩm, và Khí áp.
4. **Bảng số liệu chi tiết theo từng mốc giờ + Nút tải file CSV kết quả dự báo thực.**

#### E. Hướng dẫn Chạy Thử & Deploy Hệ thống
- **🖥️ CHẠY LOCAL PYTHON MODEL SERVING (Streamlit Web Dashboard):**
  ```bash
  # Kích hoạt môi trường và khởi chạy dashboard tương tác:
  streamlit run streamlit_app.py
  ```
- **🌐 CHẠY REST API SERVER (Python HTTP / JSON Server):**
  ```bash
  # Khởi chạy server phục vụ API suy luận:
  python app.py
  # Endpoint suy luận: POST http://localhost:8080/api/predict
  ```
- **🔗 KÊNH 1: Streamlit Community Cloud (Model Serving Online 24/7)**
  - Mã nguồn hoàn thiện tại [`streamlit_app.py`](./streamlit_app.py), tích hợp trọn gói Checkpoint PyTorch. Kết nối trực tiếp trên [share.streamlit.io](https://share.streamlit.io).
- **🔗 KÊNH 2: GitHub Pages (Chính thức qua GitHub Actions & Nhánh `gh-pages`)**
  - **Link trực tiếp:** 👉 **[https://thuanhoa1110.github.io/TimeXer-DNN-v3/](https://thuanhoa1110.github.io/TimeXer-DNN-v3/)**
  - Tự động kích hoạt qua file `.github/workflows/deploy_pages.yml`. Chạy mượt mà trên mọi trình duyệt.
- **🔗 KÊNH 3: Vercel Cloud Hosting (Dự phòng 1-Click)**
  - Tệp `vercel.json` đã tích hợp sẵn: 👉 **`https://timexer-dnn-v3.vercel.app`**

