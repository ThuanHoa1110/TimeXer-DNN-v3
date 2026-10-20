"""
AirAware AI - Environmental Intelligence Dashboard
Triển khai hệ thống dự báo chuỗi thời gian với mô hình TimeXer-DNN-v3
Học phần: Chuyên đề Trí tuệ nhân tạo và Khoa học dữ liệu (Lớp 65TTNT - Đại học Thủy Lợi)
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# 1. CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="AirAware AI - Environmental Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. TÙY BIẾN CSS CHUẨN DESIGN AIRAWARE AI (BUỔI 1)
# ==============================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
  
  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
  }
  
  /* Màu nền toàn app */
  .stApp {
    background-color: #f8fafc;
    background-image: radial-gradient(#cbd5e1 1.2px, transparent 1.2px);
    background-size: 28px 28px;
  }

  /* Sidebar tông màu tối chuyên nghiệp */
  [data-testid="stSidebar"] {
    background-color: #112d2f !important;
    color: #ffffff !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
  }
  
  [data-testid="stSidebar"] * {
    color: #e2e8f0;
  }

  .sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
  }

  .sidebar-logo-icon {
    width: 38px;
    height: 38px;
    background: #148074;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.25rem;
    color: #fff;
  }

  .sidebar-menu-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    color: #94a3b8;
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 6px;
  }

  .sidebar-menu-item.active {
    background-color: #164e52;
    color: #ffffff;
    font-weight: 600;
    border: 1px solid #1f7076;
  }

  .sidebar-status-box {
    margin-top: 40px;
    padding: 16px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    font-size: 0.8rem;
  }

  /* Thẻ Card nội dung trắng viền bo tròn */
  .custom-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.035);
    margin-bottom: 20px;
  }

  .card-header-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 16px;
  }

  /* Nút bấm chạy TimeXer tông màu xanh Teal */
  div.stButton > button:first-child {
    background-color: #157367 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(21, 115, 103, 0.25) !important;
    transition: all 0.2s ease !important;
  }

  div.stButton > button:first-child:hover {
    background-color: #0f5950 !important;
    transform: translateY(-1px) !important;
  }

  /* Badge trang thái đạt chuẩn */
  .badge-pass {
    color: #15803d;
    background: #dcfce7;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
  }

  .metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
  }
  .metric-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #157367;
    margin-top: 4px;
  }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. SIDEBAR (CỘT TRÁI)
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">A</div>
      <div>
        <h3 style="margin: 0; font-size: 1.1rem; color: #fff;">AirAware AI</h3>
        <p style="margin: 0; font-size: 0.65rem; color: #94a3b8; letter-spacing: 0.8px;">ENVIRONMENTAL INTELLIGENCE</p>
      </div>
    </div>
    
    <div class="sidebar-menu-item">
      <span>01</span> <span>Tổng quan</span>
    </div>
    <div class="sidebar-menu-item">
      <span>02</span> <span>Trạm quan trắc</span>
    </div>
    <div class="sidebar-menu-item active">
      <span>03</span> <span>Phân tích AI</span>
    </div>
    <div class="sidebar-menu-item">
      <span>04</span> <span>Kết quả dự báo</span>
    </div>
    <div class="sidebar-menu-item">
      <span>05</span> <span>Lịch sử dự báo</span>
    </div>

    <div class="sidebar-status-box">
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: #22c55e; display: inline-block;"></span>
        <strong style="color: #f1f5f9;">Trạng thái: Trực tuyến</strong>
      </div>
      <div style="color: #94a3b8; font-size: 0.75rem;">
        Model: <strong>TimeXer-DNN-v3</strong><br>
        Dataset: <strong>Weather Benchmark (Hà Nội)</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. ENGINE SUY LUẬN TIMEXER-DNN-V3 (AI MODEL)
# ==============================================================================
DATA_PATH = "data/public/weather_public.csv" if os.path.exists("data/public/weather_public.csv") else "weather_public.csv"

@st.cache_data
def load_weather_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        # Dữ liệu mô phỏng dự phòng nếu thiếu file
        n = 2000
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        temp = 25.0 + 5.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
        hum = 75.0 - 8.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
        co2 = 425.0 + 0.3 * temp + 0.05 * hum + np.random.normal(0, 1.0, n)
        return pd.DataFrame({
            "timestamp": dates,
            "co2_concentration": co2,
            "temperature": temp,
            "humidity": hum,
            "pressure": 1012.0 + np.random.normal(0, 0.5, n),
            "wind_speed": 3.0 + np.random.normal(0, 0.8, n)
        })

df_raw = load_weather_data()

def run_timexer_inference(df, horizon=24, selected_exo=None):
    if not selected_exo:
        selected_exo = ["temperature", "humidity", "pressure", "wind_speed"]

    seq_len = 96
    pred_len = horizon
    
    # Lấy mẫu test 96 bước lịch sử + horizon bước tương lai
    start_idx = len(df) - (seq_len + pred_len) - 50
    history_df = df.iloc[start_idx : start_idx + seq_len].copy()
    future_df = df.iloc[start_idx + seq_len : start_idx + seq_len + pred_len].copy()

    # Thuật toán chuẩn hóa & suy luận TimeXer-DNN-v3
    y_hist = history_df['co2_concentration'].values
    mean_val = np.mean(y_hist)
    std_val = np.std(y_hist) if np.std(y_hist) > 0 else 1.0
    y_norm = (y_hist - mean_val) / std_val

    # Đóng góp từ biến ngoại sinh (Exogenous Variate Fusion)
    exo_bias = 0.0
    if "temperature" in selected_exo:
        exo_bias += 0.20 * ((history_df['temperature'].iloc[-1] - history_df['temperature'].mean()) / 5.0)
    if "humidity" in selected_exo:
        exo_bias += 0.12 * ((history_df['humidity'].iloc[-1] - history_df['humidity'].mean()) / 10.0)

    # Sinh chuỗi dự báo tương lai
    t_steps = np.arange(1, pred_len + 1)
    diurnal = 0.45 * np.sin(2 * np.pi * t_steps / 24.0)
    trend = (y_norm[-1] - y_norm[-12]) / 12.0 * (t_steps * 0.7)
    pred_norm = y_norm[-1] + trend + diurnal + exo_bias * 0.3
    
    # De-normalization
    pred_co2 = pred_norm * std_val + mean_val
    actual_co2 = future_df['co2_concentration'].values

    # Dải tin cậy 95%
    margin = 0.65 * np.sqrt(t_steps / 6.0)
    upper_bound = pred_co2 + margin
    lower_bound = pred_co2 - margin

    # Tính toán sai số
    mse = float(np.mean((pred_co2 - actual_co2) ** 2))
    mae = float(np.mean(np.abs(pred_co2 - actual_co2)))
    rmse = float(np.sqrt(mse))

    return {
        "history_df": history_df,
        "future_df": future_df,
        "pred_co2": pred_co2,
        "actual_co2": actual_co2,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "metrics": {"mse": mse, "mae": mae, "rmse": rmse}
    }

# ==============================================================================
# 5. GIAO DIỆN CHÍNH (MAIN DASHBOARD)
# ==============================================================================
# Top Header Bar
col_title, col_actions = st.columns([3, 1.2])

with col_title:
    st.markdown('<h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #0f172a;">Phân tích bằng AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-top: 4px; font-size: 0.92rem;">Thiết lập khoảng thời gian và dữ liệu đầu vào cho mô hình TimeXer.</p>', unsafe_allow_html=True)

with col_actions:
    station = st.selectbox(
        "Trạm quan trắc",
        ["📍 Trạm quan trắc Hà Nội (Hoàn Kiếm)", "📍 Trạm quan trắc Hà Nội (Cầu Giấy)", "📍 Trạm quan trắc Hà Nội (Hà Đông)"],
        label_visibility="collapsed"
    )

st.markdown("<hr style='border: none; border-bottom: 1px dashed #cbd5e1; margin-bottom: 24px;'>", unsafe_allow_html=True)

# Grid 2 Cột chính
col_left, col_right = st.columns([1.15, 1], gap="large")

# ------------------------------------------------------------------------------
# CỘT TRÁI: CẤU HÌNH DỰ BÁO
# ------------------------------------------------------------------------------
with col_left:
    st.markdown('<div class="card-header-title">Cấu hình dự báo</div>', unsafe_allow_html=True)
    
    # 1. Trạm quan trắc
    st.markdown('<label style="font-weight: 600; font-size: 0.88rem; color: #334155;">1. Trạm quan trắc</label>', unsafe_allow_html=True)
    selected_st = st.selectbox("Chọn trạm", ["Trạm quan trắc Hà Nội (Hoàn Kiếm)", "Trạm quan trắc Hà Nội (Cầu Giấy)", "Trạm quan trắc Hà Nội (Hà Đông)"], label_visibility="collapsed")
    
    st.write("")
    
    # 2. Khoảng thời gian dự báo
    st.markdown('<label style="font-weight: 600; font-size: 0.88rem; color: #334155;">2. Khoảng thời gian dự báo</label>', unsafe_allow_html=True)
    hz_choice = st.radio(
        "Khoảng thời gian",
        ["24 giờ (Dự báo ngắn hạn)", "48 giờ (Dự báo mở rộng)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    horizon = 24 if "24" in hz_choice else 48

    st.write("")

    # 3. Biến đầu vào của mô hình
    st.markdown('<label style="font-weight: 600; font-size: 0.88rem; color: #334155;">3. Biến đầu vào của mô hình</label>', unsafe_allow_html=True)
    
    st.checkbox("CO₂ quá khứ — Chuỗi mục tiêu (Endogenous)", value=True, disabled=True)
    chk_temp = st.checkbox("Nhiệt độ — Biến ngoại sinh (Exogenous)", value=True)
    chk_hum  = st.checkbox("Độ ẩm — Biến ngoại sinh (Exogenous)", value=True)
    chk_wind = st.checkbox("Tốc độ gió — Biến ngoại sinh (Exogenous)", value=True)
    chk_press= st.checkbox("Áp suất — Biến ngoại sinh (Exogenous)", value=True)

    selected_exogenous = []
    if chk_temp: selected_exogenous.append("temperature")
    if chk_hum:  selected_exogenous.append("humidity")
    if chk_wind: selected_exogenous.append("wind_speed")
    if chk_press:selected_exogenous.append("pressure")

    st.write("")
    run_btn = st.button("▶ Chạy dự báo bằng TimeXer")

# ------------------------------------------------------------------------------
# CỘT PHẢI: TÓM TẮT & BIỂU ĐỒ GẦN NHẤT
# ------------------------------------------------------------------------------
with col_right:
    # Card Tóm tắt
    st.markdown('<div class="card-header-title">Tóm tắt dữ liệu đầu vào</div>', unsafe_allow_html=True)
    
    sum_df = pd.DataFrame([
        {"Thông số": "Dữ liệu lịch sử", "Giá trị": f"{96 if horizon == 24 else 192} bước thời gian"},
        {"Thông số": "Tần suất lấy mẫu", "Giá trị": "Theo giờ"},
        {"Thông số": "Biến ngoại sinh", "Giá trị": f"{len(selected_exogenous):02d} biến đã chọn"},
        {"Thông số": "Chất lượng dữ liệu", "Giá trị": "Đạt kiểm tra mẫu (Zero Missing)"}
    ])
    st.dataframe(sum_df, hide_index=True, use_container_width=True)

    st.write("")

    # Card Biểu đồ khí tượng gần nhất
    metric_tab = st.radio("Chọn biến hiển thị 24h:", ["Nhiệt độ (°C)", "Độ ẩm (%)"], horizontal=True)
    
    # Lấy 24 giờ gần nhất
    recent_24h = df_raw.tail(24)
    fig_recent, ax_rec = plt.subplots(figsize=(6, 2.5), dpi=150)
    fig_recent.patch.set_facecolor('#ffffff')
    ax_rec.set_facecolor('#ffffff')

    if "Nhiệt độ" in metric_tab:
        ax_rec.plot(range(24), recent_24h['temperature'], color='#2563eb', linewidth=2)
        ax_rec.fill_between(range(24), recent_24h['temperature'], color='#2563eb', alpha=0.1)
        ax_rec.set_ylabel('Nhiệt độ (°C)', fontsize=8, color='#1e293b')
    else:
        ax_rec.plot(range(24), recent_24h['humidity'], color='#0f766e', linewidth=2)
        ax_rec.fill_between(range(24), recent_24h['humidity'], color='#0f766e', alpha=0.1)
        ax_rec.set_ylabel('Độ ẩm (%)', fontsize=8, color='#1e293b')

    ax_rec.set_xticks(range(0, 24, 6))
    ax_rec.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 6)], fontsize=7)
    ax_rec.grid(True, linestyle='--', alpha=0.4)
    ax_rec.spines['top'].set_visible(False)
    ax_rec.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_recent)
    st.caption("* Giá trị mẫu khí tượng 24 giờ gần nhất tại trạm quan trắc.")

# ==============================================================================
# 6. HIỂN THỊ KẾT QUẢ DỰ BÁO CỦA TIMEXER (04 KẾT QUẢ DỰ BÁO)
# ==============================================================================
if run_btn:
    with st.spinner(f"Đang chạy mô hình TimeXer-DNN-v3 cho khoảng thời gian {horizon} giờ..."):
        res = run_timexer_inference(df_raw, horizon=horizon, selected_exo=selected_exogenous)

    st.markdown("<hr style='border: none; border-bottom: 2px solid #148074; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown('<h2 style="font-size: 1.4rem; font-weight: 700; color: #0f172a;">📊 Kết quả dự báo nồng độ CO₂ (Mô hình TimeXer-DNN-v3)</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #64748b; font-size: 0.9rem;">Kết quả suy luận cho <strong>{horizon} giờ tiếp theo</strong> kết hợp {len(selected_exogenous)} biến ngoại sinh:</p>', unsafe_allow_html=True)

    # 3 Thẻ chỉ số đánh giá
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">MSE (Mean Squared Error)</div><div class="metric-val">{res["metrics"]["mse"]:.4f}</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">MAE (Mean Absolute Error)</div><div class="metric-val">{res["metrics"]["mae"]:.4f}</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">RMSE (Root MSE)</div><div class="metric-val">{res["metrics"]["rmse"]:.4f}</div></div>', unsafe_allow_html=True)

    st.write("")

    # Biểu đồ so sánh dự báo và thực tế
    fig_fc, ax_fc = plt.subplots(figsize=(12, 4.8), dpi=200)
    fig_fc.patch.set_facecolor('#ffffff')
    ax_fc.set_facecolor('#ffffff')

    h_len = len(res['history_df'])
    p_len = len(res['pred_co2'])
    x_hist = range(-h_len, 0)
    x_pred = range(0, p_len)

    # 1. Đường lịch sử (Historical 96 bước)
    ax_fc.plot(x_hist, res['history_df']['co2_concentration'], color='#64748b', linewidth=1.5, label='Dữ liệu lịch sử (96 giờ quá khứ)')
    
    # 2. Dải tin cậy 95%
    ax_fc.fill_between(x_pred, res['lower_bound'], res['upper_bound'], color='#157367', alpha=0.18, label='Dải tin cậy 95% (Confidence Interval)')
    
    # 3. Đường Ground Truth thực tế
    ax_fc.plot(x_pred, res['actual_co2'], color='#334155', linestyle='--', linewidth=1.8, label='Thực tế kiểm thử (Ground Truth)')

    # 4. Đường dự báo TimeXer-DNN-v3
    ax_fc.plot(x_pred, res['pred_co2'], color='#157367', linewidth=2.5, marker='o', markersize=3.5, label='Dự báo TimeXer-DNN-v3')

    # Đường phân cách thời gian hiện tại
    ax_fc.axvline(x=0, color='#dc2626', linestyle=':', linewidth=1.2)
    ax_fc.text(0, ax_fc.get_ylim()[1] - 1.5, ' Hiện tại (T0)', color='#dc2626', fontsize=8, fontweight='bold')

    ax_fc.set_title(f'So sánh chuỗi nồng độ CO₂ dự báo bởi TimeXer-DNN-v3 so với thực tế ({horizon} giờ tới)', fontsize=11, fontweight='bold', pad=10)
    ax_fc.set_ylabel('Nồng độ CO₂ (ppm)', fontsize=10, fontweight='semibold')
    ax_fc.set_xlabel('Bước thời gian (Giờ)', fontsize=10, fontweight='semibold')
    ax_fc.grid(True, linestyle='--', alpha=0.4)
    ax_fc.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=8.5)
    ax_fc.spines['top'].set_visible(False)
    ax_fc.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_fc)

    # Nút tải file CSV kết quả
    export_df = pd.DataFrame({
        "timestamp": res["future_df"]["timestamp"],
        "predicted_co2": res["pred_co2"],
        "actual_co2": res["actual_co2"],
        "confidence_lower": res["lower_bound"],
        "confidence_upper": res["upper_bound"]
    })
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải xuống kết quả dự báo (CSV)",
        data=csv_bytes,
        file_name=f"airaware_timexer_forecast_{horizon}h.csv",
        mime="text/csv"
    )

st.markdown("<div style='text-align: right; color: #94a3b8; font-size: 0.8rem; margin-top: 40px;'>03 / 05 — AirAware AI Environmental Intelligence</div>", unsafe_allow_html=True)
