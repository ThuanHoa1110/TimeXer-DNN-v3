import os
import sys
import time
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Thêm đường dẫn thư mục gốc vào sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from inference_engine import TimeXerInferenceEngine

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB
# ==============================================================================
st.set_page_config(
    page_title="Dự Báo Vi Khí Hậu & Thời Tiết Hà Nội",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tối ưu giao diện người dùng (Clean, Modern Weather UI)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
  }
  .stApp {
    background-color: #f8fafc;
  }
  .weather-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    text-align: center;
  }
  .weather-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .weather-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0284c7;
    margin: 6px 0 2px 0;
  }
  .weather-desc {
    font-size: 0.82rem;
    color: #10b981;
    font-weight: 600;
  }
  .alert-box {
    background: #fff7ed;
    border: 1px solid #ffedd5;
    border-left: 4px solid #f97316;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 20px;
    font-size: 0.9rem;
    color: #9a3412;
  }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. KHỞI TẠO BỘ DỰ BÁO NGẦM (INFERENCE ENGINE)
# ==============================================================================
@st.cache_resource
def load_weather_engine():
    checkpoint_file = os.path.join(BASE_DIR, "checkpoints", "timexer_dinov3_best.pth")
    return TimeXerInferenceEngine(checkpoint_path=checkpoint_file)

engine = load_weather_engine()

# Danh mục các khu vực quan trắc tại Hà Nội
LOCATIONS = {
    "all_city": {
        "name": "🌐 Toàn thành phố Hà Nội (Xem so sánh tất cả khu vực)",
        "short": "Toàn thành phố",
        "color": "#0284c7"
    },
    "st01_hoankiem": {
        "name": "📍 Hoàn Kiếm (Khu vực Phố Cổ / Lõi đô thị)",
        "short": "Hoàn Kiếm",
        "color": "#ef4444",
        "feature": "Hiệu ứng đảo nhiệt bê tông cao, giữ nhiệt ban đêm"
    },
    "st02_badinh": {
        "name": "📍 Ba Đình (Khu vực quanh Hồ Tây)",
        "short": "Ba Đình",
        "color": "#0284c7",
        "feature": "Mặt nước Hồ Tây điều hòa, ẩm cao và dịu mát hơn"
    },
    "st03_caugiay": {
        "name": "📍 Cầu Giấy (Khu đô thị mới / Mật độ cao)",
        "short": "Cầu Giấy",
        "color": "#8b5cf6",
        "feature": "Mật độ nhà cao tầng và xe cộ cao, có đỉnh nhiệt giờ tan tầm"
    },
    "st04_hadong": {
        "name": "📍 Hà Đông (Khu vực phía Tây Nam)",
        "short": "Hà Đông",
        "color": "#10b981",
        "feature": "Nhiệt độ ổn định, chu kỳ ngày đêm điển hình đồng bằng"
    },
    "st05_donganh": {
        "name": "📍 Đông Anh (Khu vực Bắc sông Hồng)",
        "short": "Đông Anh",
        "color": "#f59e0b",
        "feature": "Đồng bằng bãi bồi, thoáng gió sông Hồng"
    },
    "st06_socson": {
        "name": "📍 Sóc Sơn (Khu vực đồi gò ngoại thành)",
        "short": "Sóc Sơn",
        "color": "#06b6d4",
        "feature": "Vùng bán sơn địa, ban đêm nhiệt độ hạ sâu hơn nội thành 2-3°C"
    }
}


# ==============================================================================
# 3. BỘ ĐIỀU KHIỂN BÊN TRÁI (BỘ LỌC THỜI TIẾT DÀNH CHO NGƯỜI DÙNG)
# ==============================================================================
with st.sidebar:
    st.markdown("### 🌦️ Dự Báo Thời Tiết")
    st.markdown("---")

    # 1. Chọn địa điểm
    selected_loc_key = st.selectbox(
        "Khu vực quan tâm:",
        options=list(LOCATIONS.keys()),
        format_func=lambda x: LOCATIONS[x]["name"],
        index=0
    )

    # 2. Chọn khoảng thời gian
    horizon_choice = st.radio(
        "Khoảng thời gian dự báo:",
        options=[24, 48, 72],
        format_func=lambda x: f"{x//24 * 24} giờ tới ({x//24} ngày)",
        index=0
    )

    st.markdown("---")
    
    # Tùy chọn nạp dữ liệu riêng
    with st.expander("📤 Tải lên dữ liệu trạm đo riêng (Tùy chọn)"):
        uploaded_files = st.file_uploader(
            "Chọn tệp CSV dữ liệu thời tiết:",
            type=["csv"],
            accept_multiple_files=True
        )


# ==============================================================================
# 4. XỬ LÝ DỮ LIỆU & CHẠY DỰ BÁO NGẦM
# ==============================================================================
st.title("🌦️ Giám Sát & Dự Báo Vi Khí Hậu Hà Nội")

# Thu thập dữ liệu các trạm
stations_data = {}
lookback_len = 96
horizon = horizon_choice

if uploaded_files:
    for f in uploaded_files:
        try:
            df_up = pd.read_csv(f)
            st_key = f.name.replace(".csv", "")
            stations_data[st_key] = df_up
        except Exception:
            pass

if not stations_data:
    if selected_loc_key == "all_city":
        for k in list(LOCATIONS.keys())[1:]:
            p = os.path.join(BASE_DIR, "data", "stations_sample", f"{k}.csv")
            if os.path.exists(p):
                df_sample = pd.read_csv(p)
                df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp'])
                stations_data[k] = df_sample.iloc[-lookback_len:].copy()
    else:
        p = os.path.join(BASE_DIR, "data", "stations_sample", f"{selected_loc_key}.csv")
        if os.path.exists(p):
            df_sample = pd.read_csv(p)
            df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp'])
            stations_data[selected_loc_key] = df_sample.iloc[-lookback_len:].copy()

# Thực hiện dự báo ngầm bằng model
forecast_results = {}
if stations_data:
    for k, df_s in stations_data.items():
        try:
            res = engine.predict(df_s, horizon=horizon, use_exogenous=True)
            forecast_results[k] = res
        except Exception:
            pass


# ==============================================================================
# 5. HIỂN THỊ THÔNG TIN THỜI TIẾT CHO NGƯỜI DÙNG
# ==============================================================================
if not forecast_results:
    st.warning("Đang tải dữ liệu thời tiết khu vực...")
else:
    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 1: XEM TOÀN THÀNH PHỐ HÀ NỘI
    # --------------------------------------------------------------------------
    if len(forecast_results) > 1:
        current_temps = [r['metrics']['current_temp'] for r in forecast_results.values()]
        max_temps = [r['metrics']['predicted_max'] for r in forecast_results.values()]
        min_temps = [r['metrics']['predicted_min'] for r in forecast_results.values()]

        uhi_diff = round(max(current_temps) - min(current_temps), 1)
        hottest_val = max(max_temps)
        coldest_val = min(min_temps)
        avg_now = round(np.mean(current_temps), 1)

        # Cảnh báo vi khí hậu tự nhiên
        if uhi_diff >= 2.0:
            st.markdown(f"""
            <div class="alert-box">
              ⚠️ <b>Cảnh báo hiệu ứng Đảo nhiệt đô thị (UHI):</b> Các quận nội thành (Hoàn Kiếm, Cầu Giấy) đang nóng hơn các khu vực ven đô (Sóc Sơn, Đông Anh) <b>+{uhi_diff}°C</b> do mật độ bê tông hóa và phương tiện giao thông.
            </div>
            """, unsafe_allow_html=True)

        # 4 Thẻ tóm tắt thông tin thời tiết
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="weather-card">
              <div class="weather-title">Nhiệt độ hiện tại (Trung bình)</div>
              <div class="weather-value">{avg_now}°C</div>
              <div class="weather-desc">Toàn địa bàn Hà Nội</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="weather-card">
              <div class="weather-title">Nóng nhất trong đợt</div>
              <div class="weather-value">{hottest_val}°C</div>
              <div class="weather-desc">Đỉnh nhiệt buổi trưa</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="weather-card">
              <div class="weather-title">Lạnh nhất ban đêm</div>
              <div class="weather-value">{coldest_val}°C</div>
              <div class="weather-desc">Thời điểm rạng sáng</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="weather-card">
              <div class="weather-title">Chênh lệch Nội thành - Ngoại ô</div>
              <div class="weather-value">+{uhi_diff}°C</div>
              <div class="weather-desc">Đảo nhiệt đô thị</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Diễn Biến Nhiệt Độ Theo Giờ Tại Các Khu Vực Hà Nội")

        # Vẽ biểu đồ thời tiết các khu vực
        fig, ax = plt.subplots(figsize=(14, 5.5), dpi=130)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#fcfcfc')

        palette = ["#ef4444", "#0284c7", "#8b5cf6", "#10b981", "#f59e0b", "#06b6d4"]
        first_res = list(forecast_results.values())[0]
        timestamps = first_res['predictions']['timestamp'].dt.strftime('%H:%M (%d/%m)').values
        
        # Thưa bớt nhãn thời gian trục X cho người dùng dễ đọc
        step_stride = max(1, len(timestamps) // 8)
        x_indices = np.arange(len(timestamps))

        for idx, (st_k, res) in enumerate(forecast_results.items()):
            temps = res['predictions']['predicted_temperature'].values
            info = LOCATIONS.get(st_k, {})
            loc_label = info.get("short", st_k)
            loc_color = info.get("color", palette[idx % len(palette)])

            ax.plot(x_indices, temps, color=loc_color, linewidth=2.4, marker='o', markersize=3, label=f"{loc_label} ({temps.min():.1f}° - {temps.max():.1f}°C)")

        ax.set_xticks(x_indices[::step_stride])
        ax.set_xticklabels(timestamps[::step_stride], rotation=0, fontsize=9)
        ax.set_ylabel("Nhiệt độ không khí (°C)", fontsize=10, labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9, ncol=2)
        plt.tight_layout()
        st.pyplot(fig)

        # Bảng tóm tắt so sánh
        st.markdown("### 📋 So Sánh Thời Tiết Giữa Các Quận / Huyện")
        table_rows = []
        for st_k, res in forecast_results.items():
            temps = res['predictions']['predicted_temperature'].values
            info = LOCATIONS.get(st_k, {})
            table_rows.append({
                "Khu Vực": info.get("name", st_k),
                "Nhiệt Độ Hiện Tại": f"{res['metrics']['current_temp']}°C",
                "Thấp Nhất Đêm": f"{temps.min():.1f}°C",
                "Cao Nhất Trưa": f"{temps.max():.1f}°C",
                "Biên Độ Ngày Đêm": f"{(temps.max() - temps.min()):.1f}°C",
                "Đặc Điểm Vi Khí Hậu": info.get("feature", "Bình thường")
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # Nút tải bảng dự báo
        csv_export = pd.DataFrame({"Thời gian": timestamps})
        for st_k, res in forecast_results.items():
            loc_short = LOCATIONS.get(st_k, {}).get("short", st_k)
            csv_export[f"Nhiệt độ {loc_short} (°C)"] = res['predictions']['predicted_temperature'].values

        csv_buf = io.StringIO()
        csv_export.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Tải Bảng Dự Báo Thời Tiết Toàn Hà Nội (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"du_bao_thoi_tiet_ha_noi_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 2: XEM 1 KHU VỰC CỤ THỂ
    # --------------------------------------------------------------------------
    else:
        st_k = list(forecast_results.keys())[0]
        res = forecast_results[st_k]
        pred_df = res['predictions']
        history_df = res['lookback_history']
        m = res['metrics']
        loc_info = LOCATIONS.get(st_k, {})
        loc_name = loc_info.get("name", st_k)

        # 4 Thẻ thời tiết cho khu vực
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="weather-card">
              <div class="weather-title">Nhiệt độ hiện tại</div>
              <div class="weather-value">{m['current_temp']}°C</div>
              <div class="weather-desc">Khu vực: {loc_info.get('short', st_k)}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="weather-card">
              <div class="weather-title">Nóng nhất buổi trưa</div>
              <div class="weather-value">{m['predicted_max']}°C</div>
              <div class="weather-desc">Đỉnh nhiệt cao nhất</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="weather-card">
              <div class="weather-title">Hạ thấp ban đêm</div>
              <div class="weather-value">{m['predicted_min']}°C</div>
              <div class="weather-desc">Rạng sáng thấp nhất</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="weather-card">
              <div class="weather-title">Biên độ ngày đêm</div>
              <div class="weather-value">{m['amplitude']}°C</div>
              <div class="weather-desc">Độ chênh nhiệt độ</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"### 📈 Diễn Biến Nhiệt Độ Chi Tiết Tại {loc_name}")

        # Vẽ biểu đồ thời tiết chi tiết theo giờ
        fig, ax = plt.subplots(figsize=(14, 5.2), dpi=130)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#fcfcfc')

        # Dữ liệu quá khứ & tương lai
        hist_temps = history_df['temperature'].values
        pred_temps = pred_df['predicted_temperature'].values
        
        all_x = np.arange(len(hist_temps) + len(pred_temps))
        hist_x = all_x[:len(hist_temps)]
        pred_x = all_x[len(hist_temps):]

        ax.plot(hist_x, hist_temps, color='#64748b', linewidth=2.0, label='Quan sát gần đây')
        ax.plot(pred_x, pred_temps, color='#0284c7', linewidth=2.6, marker='o', markersize=4, label='Dự báo các giờ tới')

        # Đánh dấu đỉnh trưa và đáy đêm
        peak_idx = np.argmax(pred_temps)
        low_idx = np.argmin(pred_temps)
        ax.scatter([pred_x[peak_idx]], [pred_temps[peak_idx]], color='#ef4444', s=100, zorder=5, label=f'Đỉnh nhiệt ({pred_temps[peak_idx]:.1f}°C)')
        ax.scatter([pred_x[low_idx]], [pred_temps[low_idx]], color='#06b6d4', s=100, zorder=5, label=f'Đáy nhiệt ({pred_temps[low_idx]:.1f}°C)')

        ax.axvline(x=len(hist_temps) - 1, color='#f97316', linestyle=':', linewidth=1.8, label='Hiện tại')
        ax.set_ylabel("Nhiệt độ (°C)", fontsize=10, labelpad=8)
        ax.set_xlabel("Diễn biến thời gian", fontsize=10, labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        # Bảng dữ liệu từng giờ
        st.markdown("### 📋 Bảng Dự Báo Theo Mốc Thời Gian")
        detail_df = pd.DataFrame({
            "Thời Gian": pred_df['timestamp'].dt.strftime('%H:%M ngày %d/%m/%Y'),
            "Nhiệt Độ Dự Báo (°C)": pred_df['predicted_temperature'],
            "Xu Hướng": ["Đang tăng nhiệt" if i > 0 and pred_temps[i] > pred_temps[i-1] else "Đang hạ nhiệt" for i in range(len(pred_temps))]
        })
        st.dataframe(detail_df, use_container_width=True, height=220)

        # Nút tải bảng dự báo
        csv_buf = io.StringIO()
        detail_df.to_csv(csv_buf, index=False)
        st.download_button(
            label=f"📥 Tải Bảng Dự Báo Thời Tiết ({loc_info.get('short', st_k)})",
            data=csv_buf.getvalue(),
            file_name=f"du_bao_nhiet_do_{st_k}_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
