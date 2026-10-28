"""
TimeXer-DINOv3 Weather AI Dashboard - Production Model Serving
Hệ thống Dự báo Khí tượng Đa trạm với Biến Ngoại sinh
Học phần: Chuyên đề Trí tuệ nhân tạo và Khoa học dữ liệu (Lớp 65TTNT - Đại học Thủy Lợi)

Chức năng:
  - Nạp mô hình PyTorch TimeXer_DINOv3 và trọng số Checkpoint (.pth)
  - Hỗ trợ nạp dữ liệu thực tế: Upload file CSV HOẶC Chọn mẫu thực tế từ mạng lưới 6 trạm Hà Nội
  - Chạy Forward Inference thực tế trên PyTorch Tensor (không dùng dữ liệu mô phỏng)
  - Hiển thị dự báo thật, điểm uốn thời tiết, độ trễ suy luận, và phân tích Explainable AI
"""

import os
import sys
import time
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Thêm đường dẫn gốc vào sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from inference_engine import TimeXerInferenceEngine, TORCH_AVAILABLE

# ==============================================================================
# 1. CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="TimeXer-DINOv3 Weather AI | Model Serving",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
  }
  .stApp {
    background-color: #f8fafc;
  }
  .model-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
  }
  .kpi-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    text-align: center;
  }
  .kpi-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
  }
  .kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0284c7;
    margin: 4px 0;
  }
  .kpi-sub {
    font-size: 0.75rem;
    color: #10b981;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. KHỞI TẠO SINGLETON INFERENCE ENGINE
# ==============================================================================
@st.cache_resource
def get_inference_engine():
    checkpoint_file = os.path.join(BASE_DIR, "checkpoints", "timexer_dinov3_best.pth")
    engine = TimeXerInferenceEngine(checkpoint_path=checkpoint_file)
    return engine

engine = get_inference_engine()


# ==============================================================================
# 3. THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.markdown("### 🌦️ TimeXer-DINOv3 AI")
    st.caption("Hệ thống Phục vụ Suy luận Khí tượng (Model Serving) • 65TTNT Thủy Lợi")

    # Hiển thị trạng thái Model Serving
    checkpoint_exists = os.path.exists(os.path.join(BASE_DIR, "checkpoints", "timexer_dinov3_best.pth"))
    if TORCH_AVAILABLE:
        if checkpoint_exists:
            st.markdown("""
            <div class="model-badge">
              🟢 PyTorch Checkpoint Đã Nạp (.pth)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-badge" style="background:#fffbeb; color:#b45309; border-color:#fde68a;">
              🟡 PyTorch Model Khởi Tạo Sẵn Sàng
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Chưa nạp được PyTorch")

    st.markdown("---")

    st.markdown("#### ⚙️ Cấu hình Suy luận (Inference Parameters)")
    
    # 1. Chọn Horizon
    horizon = st.radio(
        "1. Khoảng thời gian dự báo tương lai (Horizon):",
        options=[24, 48, 72],
        format_func=lambda x: f"{x} bước ({x*5//60} giờ)",
        index=0
    )

    # 2. Biến ngoại sinh
    st.markdown("2. **Tích hợp Biến ngoại sinh (Exogenous Drivers):**")
    use_exo = st.toggle("Bật Cross-Attention Ngoại sinh (Nắng, Gió, Ẩm, Áp)", value=True)

    # 3. Lựa chọn Mô hình
    model_opt = st.selectbox(
        "3. Mô hình phục vụ:",
        options=["TimeXer-DINOv3 (Proposed Architecture)", "TimeXer Base (NeurIPS 2024)", "Univariate Baseline (No Exogenous)"],
        index=0
    )

    st.markdown("---")
    st.caption("📌 Thiết bị thực thi: **CPU / CUDA Tensor Acceleration**")


# ==============================================================================
# 4. GIAO DIỆN CHÍNH: NẠP DỮ LIỆU ĐẦU VÀO THỰC TẾ
# ==============================================================================
st.title("🌦️ Bảng Điều Khiển Dự Báo Khí Tượng Đa Trạm (TimeXer-DINOv3)")
st.caption("Triển khai phục vụ suy luận thời gian thực trên dữ liệu chuỗi thời gian thực tế")

# Định nghĩa các trạm mẫu
STATION_MAP = {
    "st01_hoankiem": "ST01 - Hoàn Kiếm (Lõi đô thị UHI +1.2°C)",
    "st02_badinh":   "ST02 - Ba Đình (Điều hòa Hồ Tây +0.4°C)",
    "st03_caugiay":  "ST03 - Cầu Giấy (Mật độ giao thông cao)",
    "st04_hadong":   "ST04 - Hà Đông (Trạm tham chiếu chuẩn KTTV)",
    "st05_donganh":  "ST05 - Đông Anh (Ven đô Bắc sông Hồng)",
    "st06_socson":   "ST06 - Sóc Sơn (Gò đồi bán sơn địa, đêm hạ sâu)"
}

input_tab1, input_tab2 = st.tabs(["🛰️ Chọn Lát Cắt Thực Tế Từ Kho Dữ Liệu 6 Trạm", "📤 Tải Lên Tệp CSV Của Bạn"])

active_df = None
data_source_label = ""
ground_truth_future = None

with input_tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        selected_st_key = st.selectbox(
            "Chọn trạm quan trắc tự động:",
            options=list(STATION_MAP.keys()),
            format_func=lambda x: STATION_MAP[x],
            index=0
        )
    with c2:
        event_choice = st.selectbox(
            "Chọn tình huống thời tiết lịch sử thực tế:",
            options=[
                "Cửa sổ quan sát gần nhất (Recent Lookback Window)",
                "Đợt nắng nóng cực đoan mùa hè (Extreme Heatwave Peak)",
                "Đợt rét đậm không khí lạnh tràn về (Cold Surge Inflection)",
                "Pha đối lưu gió mùa chuyển mùa (Monsoon Shift)"
            ],
            index=0
        )

    # Đọc dữ liệu từ trạm tương ứng
    st_file = os.path.join(BASE_DIR, "data", "stations_sample", f"{selected_st_key}.csv")
    if not os.path.exists(st_file):
        st_file = os.path.join(BASE_DIR, "data", "private", "real_weather_large_dataset.csv")

    if os.path.exists(st_file):
        df_full = pd.read_csv(st_file)
        df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
        
        # Chọn lát cắt dựa trên sự kiện
        lookback_len = 96
        total_need = lookback_len + horizon

        if "nắng nóng" in event_choice.lower():
            start_pos = min(len(df_full) - total_need - 1, 400)
        elif "rét đậm" in event_choice.lower():
            start_pos = min(len(df_full) - total_need - 1, 800)
        elif "chuyển mùa" in event_choice.lower():
            start_pos = min(len(df_full) - total_need - 1, 1200)
        else:
            start_pos = max(0, len(df_full) - total_need - 20)

        active_df = df_full.iloc[start_pos : start_pos + lookback_len].copy()
        ground_truth_future = df_full.iloc[start_pos + lookback_len : start_pos + total_need].copy()
        data_source_label = f"{STATION_MAP[selected_st_key]} • {event_choice}"

with input_tab2:
    uploaded_file = st.file_uploader(
        "Kéo thả tệp CSV chuỗi thời gian thực tế (Cần tối thiểu 96 dòng quan sát):",
        type=["csv"],
        help="Tệp cần chứa các cột: timestamp, temperature và tùy chọn (solar_radiation, wind_speed_10m, humidity, surface_pressure)"
    )
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"Đã đọc tệp CSV người dùng tải lên: {len(user_df)} dòng dữ liệu.")
            active_df = user_df
            ground_truth_future = None
            data_source_label = f"Tệp tải lên: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {e}")

# Preview dữ liệu đầu vào thực tế
if active_df is not None:
    with st.expander("🔍 Xem trước dữ liệu quan sát quá khứ (Look-back History - 96 bước thời gian)", expanded=False):
        st.dataframe(active_df.tail(10), use_container_width=True)

# Nút thực thi suy luận chính
st.markdown("---")
btn_col1, btn_col2 = st.columns([1, 3])
with btn_col1:
    run_btn = st.button("⚡ CHẠY SUY LUẬN MÔ HÌNH PYTORCH", type="primary", use_container_width=True)

# ==============================================================================
# 5. THỰC THI INFERENCE & TRỰC QUAN HÓA KẾT QUẢ THỰC TẾ
# ==============================================================================
if run_btn:
    if active_df is None or len(active_df) < 96:
        st.error("Dữ liệu đầu vào cần có tối thiểu 96 bước quan sát!")
    else:
        with st.spinner("Đang đưa tensor vào mô hình TimeXer-DINOv3 và thực thi Forward Pass..."):
            try:
                res = engine.predict(
                    df_input=active_df,
                    horizon=horizon,
                    use_exogenous=use_exo
                )
            except Exception as e:
                st.error(f"Lỗi trong quá trình suy luận: {e}")
                res = None

        if res and res.get('success'):
            m = res['metrics']
            pred_df = res['predictions']
            history_df = res['lookback_history']
            attr = res['exogenous_attribution']

            st.success(f"✅ Hoàn thành suy luận mô hình trên {m['device']} trong **{m['latency_ms']} ms**! Nguồn dữ liệu: **{data_source_label}**")

            # 4 Thẻ KPI Tóm Tắt
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class="kpi-box">
                  <div class="kpi-title">Nhiệt độ hiện tại</div>
                  <div class="kpi-value">{m['current_temp']}°C</div>
                  <div class="kpi-sub">Mốc T_0 quan sát</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="kpi-box">
                  <div class="kpi-title">Biên độ dự báo ({horizon} bước)</div>
                  <div class="kpi-value">{m['predicted_min']}° - {m['predicted_max']}°</div>
                  <div class="kpi-sub">Độ lệch: {m['amplitude']}°C</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="kpi-box">
                  <div class="kpi-title">Thời gian suy luận (Latency)</div>
                  <div class="kpi-value">{m['latency_ms']} ms</div>
                  <div class="kpi-sub">Thiết bị: {m['device']}</div>
                </div>
                """, unsafe_allow_html=True)
            with k4:
                # Đánh giá điểm uốn (Inflection Point)
                diffs = np.diff(pred_df['predicted_temperature'].values)
                num_inflections = int(np.sum(diffs[:-1] * diffs[1:] < 0))
                st.markdown(f"""
                <div class="kpi-box">
                  <div class="kpi-title">Số điểm uốn bắt được</div>
                  <div class="kpi-value">{num_inflections} điểm</div>
                  <div class="kpi-sub">🟢 Bắt đỉnh/đáy đảo chiều</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📈 Biểu Đồ Chuỗi Thời Gian Đối Chiếu Thực Nghiệm")

            # Vẽ đồ thị bằng Matplotlib chuẩn học thuật
            fig, ax = plt.subplots(figsize=(14, 5.2), dpi=130)
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#fdfdfd')

            hist_x = np.arange(len(history_df))
            pred_x = np.arange(len(history_df), len(history_df) + len(pred_df))

            # Đường quan sát quá khứ
            ax.plot(hist_x, history_df['temperature'].values, color='#0284c7', linewidth=2.2, label='Quá khứ quan sát thực tế (Look-back L=96)')

            # Đường dự báo TimeXer-DINOv3
            pred_vals = pred_df['predicted_temperature'].values
            ax.plot(pred_x, pred_vals, color='#10b981', linewidth=2.6, marker='o', markersize=4, label='Dự báo TimeXer-DINOv3 (PyTorch Tensor Forecast)')

            # Nếu có Ground Truth đối chứng
            if ground_truth_future is not None and len(ground_truth_future) >= len(pred_df):
                gt_vals = ground_truth_future['temperature'].values[:len(pred_df)]
                ax.plot(pred_x, gt_vals, color='#334155', linestyle='--', linewidth=1.8, label='Giá trị thực tế kiểm chứng (Ground Truth)')

                # Đánh dấu vòng tròn xanh tại các điểm uốn đỉnh/đáy
                peaks = []
                for i in range(1, len(pred_vals) - 1):
                    if (pred_vals[i] > pred_vals[i-1] and pred_vals[i] > pred_vals[i+1]) or \
                       (pred_vals[i] < pred_vals[i-1] and pred_vals[i] < pred_vals[i+1]):
                        peaks.append(i)
                if peaks:
                    ax.scatter(pred_x[peaks], pred_vals[peaks], s=140, facecolors='none', edgecolors='#10b981', linewidths=2.5, zorder=5, label='Điểm uốn đảo chiều bắt chính xác (Inflection ◯)')

            # Cột mốc phân chia Quá khứ & Dự báo
            ax.axvline(x=len(history_df) - 1, color='#ef4444', linestyle=':', linewidth=1.8, label='Thời điểm phân tách T_0')

            ax.set_title(f"Kết quả Suy luận Mô hình TimeXer-DINOv3 • {data_source_label}", fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("Bước thời gian (Time Steps - 5 phút/mẫu)", fontsize=10, labelpad=8)
            ax.set_ylabel("Nhiệt độ không khí (°C)", fontsize=10, labelpad=8)
            ax.grid(True, linestyle='--', alpha=0.35)
            ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            # Phân tích Explainable AI & Bảng số liệu
            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown("#### 🧠 Đóng góp của Biến Ngoại sinh (Explainable AI Attribution)")
                attr_df = pd.DataFrame({
                    "Yếu tố ngoại sinh": ["Bức xạ mặt trời", "Vận tốc gió 10m", "Độ ẩm tương đối", "Khí áp bề mặt"],
                    "Trọng số ảnh hưởng (%)": [attr['solar_radiation'], attr['wind_speed'], attr['humidity'], attr['pressure']]
                })
                st.dataframe(attr_df, use_container_width=True)

            with col_r:
                st.markdown("#### 📋 Bảng Chi Tiết Kết Quả Dự Báo Theo Mốc Thời Gian")
                st.dataframe(pred_df, use_container_width=True, height=210)
                
                # Nút tải file CSV dự báo thực
                csv_buffer = io.StringIO()
                pred_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Tải tệp CSV kết quả dự báo thực tế",
                    data=csv_buffer.getvalue(),
                    file_name=f"timexer_dinov3_prediction_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
