# 📌 BÁO CÁO TỔNG HỢP 3 BÀI NỘP - HỌC PHẦN CHUYÊN ĐỀ TTNT & KHDL (65TTNT)
> **Trường Đại học Thủy Lợi - Khoa Công nghệ Thông tin**  
> **Sinh viên:** Nguyễn Thuận Hóa  
> **Repository:** https://github.com/ThuanHoa1110/TimeXer-DNN-v3.git  

---

## 📋 TỔNG HỢP NỘI DUNG 3 BÀI THEO YÊU CẦU CỦA THẦY TRÊN BẢNG

```
┌────────────────────────────────────────────────────────────────────────┐
│ YÊU CẦU TRÊN BẢNG CỦA THẦY:                                            │
│ 1. Code thay encoder (hoặc backbone) bằng DNN-v3.                       │
│    -> Submit GitHub kèm README chỉ chỗ đã update.                      │
│ 2. Thu thập data private theo đúng luồng hôm trước.                    │
│    -> Submit link Kaggle public.                                       │
│ 3. Code design như buổi 1 kèm model AI đã xuất.                        │
│    -> Link deploy hệ thống.                                            │
│ => Tổng thể: Submit file PDF kèm 3 bài.                                │
└────────────────────────────────────────────────────────────────────────┘
```

---

### BÀI 1: Thay Encoder Backbone bằng mạng DNN-v3
- **Link GitHub Repository:** https://github.com/ThuanHoa1110/TimeXer-DNN-v3
- **Các file đã code & cập nhật:**
  1. `models/TimeXer_DnnV3.py`: Triển khai độc lập kiến trúc mạng nơ-ron sâu 3 giai đoạn (DNN-v3) thay thế Attention Encoder của TimeXer:
     - *Stage 1:* Intra-patch Temporal Feature Representation MLP
     - *Stage 2:* Cross-Variate Exogenous Fusion MLP (Tích hợp thông tin ngoại sinh)
     - *Stage 3:* Deep Feed-Forward Expansion & Residual LayerNorm
  2. `models/TimeXer.py`: Bổ sung tham số `--encoder_type dnn_v3` vào lớp Model chính.
  3. `run_test_dnn_v3.py`: Unit test kiểm tra tensor shape đầu vào/đầu ra cho cả 2 nhiệm vụ MS (Multivariate predict Univariate) và M (Multivariate).
  4. `README.md`: Báo cáo chi tiết vị trí các dòng code đã chỉnh sửa và biểu đồ kiến trúc tensor.

---

### BÀI 2: Thu thập Dataset Private & Publish Kaggle
- **Đề tài:** Private Face Biometrics Time Series Dataset for TimeXer (Dự báo góc quay đầu `head_yaw` kết hợp các biến ngoại sinh sinh trắc học khuôn mặt).
- **Đặc tả tập dữ liệu (`private_dataset.csv`):**
  - Số lượng bản ghi: 15,000 dòng liên tục (Tần số 5.0 Hz, 200ms/bước).
  - Biến nội sinh (Target): `head_yaw` (Góc quay đầu theo phương ngang, độ).
  - Biến ngoại sinh (Exogenous Covariates):
    + `ambient_light`: Cường độ ánh sáng môi trường.
    + `eye_aspect_ratio` (EAR): Chỉ số mở mắt (chỉ báo mức độ tỉnh táo/mệt mỏi).
    + `mouth_open_ratio` (MOR): Chỉ số mở miệng (nói chuyện/ngáp).
- **Bộ tài liệu Kaggle chuẩn Usability 10.0:** File `KAGGLE_DESCRIPTION.txt` và bộ 5 biểu đồ EDA chất lượng cao trong thư mục `figures/private_01_*.png` đến `private_05_*.png`.

---

### BÀI 3: Code Giao diện Design Buổi 1 (AirAware AI) kèm Model AI đã xuất & Link Deploy
- **Thiết kế:** Chuẩn 100% bản wireframe Buổi 1 **AirAware AI - Environmental Intelligence** (Màn hình 03 Phân tích AI).
- **Kiến trúc hệ thống:**
  - **Frontend:** `index.html` và `cau3_airaware.html`
    + Sidebar chuẩn tông màu trầm với chỉ mục 01 - 05 (Mục 03 Active).
    + Khung cấu hình trạm quan trắc Hà Nội, lựa chọn 24h / 48h, danh sách checkbox biến mục tiêu CO₂ và 4 biến ngoại sinh (Nhiệt độ, Độ ẩm, Tốc độ gió, Áp suất).
    + Bảng tóm tắt tự động cập nhật số lượng biến.
    + Canvas đồ thị khí tượng 24h gần nhất hỗ trợ chuyển đổi giữa Nhiệt độ và Độ ẩm.
    + Khung kết quả dự báo hiển thị đường cong CO₂ của TimeXer, đường thực tế Ground Truth và 3 chỉ số đánh giá MSE, MAE, RMSE.
  - **Backend & AI Inference Engine:** `app.py`
    + Máy chủ web độc lập chạy cổng `8080` (hỗ trợ đọc cổng từ biến môi trường `PORT` khi deploy lên Cloud).
    + Endpoint API `POST /api/predict`: Nhận cấu hình từ giao diện, nạp dữ liệu chuỗi thời gian khí tượng thực tế (`data/public/weather_public.csv`), thực thi thuật toán dự báo TimeXer-DNN-v3 và trả về chuỗi dự báo kèm dải tin cậy 95%.
    + Endpoint API `GET /api/status`: Báo cáo tình trạng hoạt động và phiên bản mô hình AI.

---

## 🚀 HƯỚNG DẪN LẤY LINK DEPLOY ĐỂ NỘP BÀI

### Cách 1: Tạo Link Deploy Online miễn phí qua Render.com (Khuyên dùng)
1. Đẩy code lên GitHub:
   ```bash
   git add .
   git commit -m "Deploy Câu 3: AirAware AI Dashboard voi mo hinh TimeXer-DNN-v3"
   git push origin main
   ```
2. Truy cập [render.com](https://render.com), đăng nhập bằng tài khoản GitHub.
3. Chọn **New +** $\rightarrow$ **Web Service** $\rightarrow$ Chọn repo `ThuanHoa1110/TimeXer-DNN-v3`.
4. Điền cấu hình:
   - **Name:** `airaware-timexer-65ttnt`
   - **Environment:** `Python 3`
   - **Build Command:** *(để trống hoặc `pip install -r requirements.txt`)*
   - **Start Command:** `python app.py`
5. Nhấn **Create Web Service**. Sau 1-2 phút, bạn sẽ nhận được Link Deploy chính thức dạng:
   👉 **`https://airaware-timexer-65ttnt.onrender.com`**

### Cách 2: Tạo Link Deploy tức thì qua Localtunnel (Chạy ngay từ máy)
Nếu cần gửi link cho thầy chấm bài ngay lập tức khi máy đang bật:
```powershell
# Chạy máy chủ app.py
python app.py

# Ở một cửa sổ terminal khác, gõ:
npx localtunnel --port 8080
```
Hệ thống sẽ cung cấp ngay một đường link công khai (ví dụ: `https://fair-river-92.loca.lt`) có thể mở được trên mọi điện thoại/máy tính của thầy!

### Cách 3: Link GitHub Pages
Bạn có thể bật tính năng GitHub Pages trong mục **Settings $\rightarrow$ Pages** của repository để có thêm đường link tĩnh:
👉 **`https://ThuanHoa1110.github.io/TimeXer-DNN-v3`**
