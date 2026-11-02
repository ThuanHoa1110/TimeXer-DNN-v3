# ==============================================================================
# DOCKERFILE CHO BÀI TOÁN AI: TimeXer-DINOv3 Weather AI Model Serving
# Học phần: Chuyên đề Trí tuệ Nhân tạo và Khoa học Dữ liệu (Lớp 65TTNT - ĐH Thủy Lợi)
# Sinh viên thực hiện: Nguyễn Thuận Hóa
# ==============================================================================

# 1. Base Image: Python 3.10 Slim tối ưu dung lượng (chỉ ~150MB so với ~1GB của image đầy đủ)
FROM python:3.10-slim

# Thiết lập thông tin nhãn metadata
LABEL maintainer="Nguyen Thuan Hoa <thuanhoa1110@gmail.com>"
LABEL description="TimeXer-DINOv3 Deep Learning Weather Forecasting Production Serving"
LABEL version="1.0.0"

# Thiết lập biến môi trường runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 2. Cài đặt các gói hệ thống cần thiết (curl phục vụ container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# 3. Tận dụng Docker Layer Caching: Cài đặt Dependencies trước
COPY requirements.txt /app/requirements.txt

# Cài đặt PyTorch CPU-optimized wheel trước (tiết kiệm ~2GB so với bản CUDA đầy đủ khi chạy CPU serving)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện AI & Web còn lại
RUN pip install --no-cache-dir -r requirements.txt

# 4. Sao chép toàn bộ mã nguồn mô hình, checkpoint trọng số và dữ liệu trạm
COPY . /app

# 5. Khai báo cổng lắng nghe
# Port 8080: Web API REST Server (app.py)
# Port 8501: Streamlit Interactive Dashboard (streamlit_app.py)
EXPOSE 8080 8501

# 6. Container Healthcheck giám sát trạng thái sống của Model Serving
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/api/model-info || exit 1

# 7. Lệnh khởi chạy mặc định: Khởi động REST API & Model Serving
CMD ["python", "app.py"]
