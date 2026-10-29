import os
import sys
import time
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pydeck as pdk

# Cấu hình đường dẫn thư mục gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from inference_engine import TimeXerInferenceEngine

# ==============================================================================
# 1. THIẾT LẬP CẤU HÌNH & GIAO DIỆN (COMMERCIAL DESIGN SYSTEM)
# ==============================================================================
st.set_page_config(
    page_title="METEO-HANOI™ | Nền Tảng Dự Báo Vi Khí Hậu Thông Minh",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling lấy cảm hứng từ Tomorrow.io & Apple Weather
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  
  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  
  /* Background & Base Container */
  .stApp {
    background-color: #0b0f19;
    color: #f1f5f9;
  }
  
  /* Top Executive Header */
  .executive-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .brand-title {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  
  .brand-subtitle {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 4px;
  }
  
  .status-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #34d399;
  }
  
  .status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 10px #10b981;
  }

  /* Operational Advisory Banner (Tomorrow.io Style) */
  .advisory-card {
    background: linear-gradient(135deg, rgba(234, 88, 12, 0.15) 0%, rgba(249, 115, 22, 0.05) 100%);
    border: 1px solid rgba(249, 115, 22, 0.3);
    border-left: 5px solid #f97316;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 24px;
    backdrop-filter: blur(8px);
  }
  
  .advisory-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #fdba74;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    margin-bottom: 6px;
  }
  
  .advisory-content {
    font-size: 0.95rem;
    color: #fed7aa;
    line-height: 1.5;
  }

  /* Metric KPI Tiles */
  .kpi-tile {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: left;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .kpi-tile:hover {
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-2px);
  }
  .kpi-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  .kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 6px 0 2px 0;
  }
  .kpi-subtext {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
  }

  /* Hourly Ribbon Card (Apple Weather Style) */
  .hourly-container {
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding: 8px 4px 16px 4px;
    scrollbar-width: thin;
  }
  .hourly-card {
    min-width: 82px;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 10px;
    text-align: center;
    flex-shrink: 0;
  }
  .hourly-time {
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 600;
  }
  .hourly-icon {
    font-size: 1.4rem;
    margin: 6px 0;
  }
  .hourly-temp {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
  }

  /* Temperature Range Bar */
  .temp-bar-container {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .station-name-col {
    width: 220px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #e2e8f0;
  }
  .bar-wrapper {
    flex-grow: 1;
    margin: 0 20px;
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 9999px;
    position: relative;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #38bdf8 0%, #f59e0b 50%, #ef4444 100%);
  }

  /* Section Containers */
  .content-section {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 24px;
  }
  .section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. KHỞI TẠO BỘ SUY DIỄN THỜI TIẾT (INFERENCE ENGINE)
# ==============================================================================
@st.cache_resource
def load_weather_engine():
    checkpoint_file = os.path.join(BASE_DIR, "checkpoints", "timexer_dinov3_best.pth")
    return TimeXerInferenceEngine(checkpoint_path=checkpoint_file)

engine = load_weather_engine()

# Metadata hệ thống trạm quan trắc chuẩn thực tế Hà Nội
STATIONS_METADATA = {
    "all_city": {
        "id": "ALL_CITY",
        "name": "Toàn Thành Phố Hà Nội (Tổng Quan)",
        "district": "Toàn thành phố",
        "lat": 21.0285,
        "lon": 105.8542,
        "color": [14, 165, 233, 200],
        "hex": "#0ea5e9",
        "feature": "Mạng lưới tích hợp đa trạm nội - ngoại thành",
        "zone_type": "Mạng lưới đô thị thông minh"
    },
    "st01_hoankiem": {
        "id": "ST01_HOANKIEM",
        "name": "Hoàn Kiếm (Lõi Đô Thị - Phố Cổ)",
        "district": "Quận Hoàn Kiếm",
        "lat": 21.0285,
        "lon": 105.8542,
        "color": [239, 68, 68, 220],
        "hex": "#ef4444",
        "feature": "Hiệu ứng đảo nhiệt bê tông cao, giữ nhiệt ban đêm",
        "zone_type": "Lõi đô thị trung tâm"
    },
    "st02_badinh": {
        "id": "ST02_BADINH",
        "name": "Ba Đình (Quanh Hồ Tây)",
        "district": "Quận Ba Đình",
        "lat": 21.0341,
        "lon": 105.8342,
        "color": [14, 165, 233, 220],
        "hex": "#0ea5e9",
        "feature": "Mặt nước Hồ Tây điều hòa nhiệt, độ ẩm cao",
        "zone_type": "Đô thị ven hồ điều hòa"
    },
    "st03_caugiay": {
        "id": "ST03_CAUGIAY",
        "name": "Cầu Giấy (Khu Đô Thị Mới)",
        "district": "Quận Cầu Giấy",
        "lat": 21.0366,
        "lon": 105.7905,
        "color": [139, 92, 246, 220],
        "hex": "#8b5cf6",
        "feature": "Mật độ cao ốc & phương tiện lớn, đỉnh nhiệt kéo dài",
        "zone_type": "Khu đô thị hành chính"
    },
    "st04_hadong": {
        "id": "ST04_HADONG",
        "name": "Hà Đông (Khu Phía Tây Nam)",
        "district": "Quận Hà Đông",
        "lat": 20.9723,
        "lon": 105.7766,
        "color": [16, 185, 129, 220],
        "hex": "#10b981",
        "feature": "Vùng chuyển tiếp đồng bằng ven sông Đáy",
        "zone_type": "Đô thị vệ tinh"
    },
    "st05_donganh": {
        "id": "ST05_DONGANH",
        "name": "Đông Anh (Bắc Sông Hồng)",
        "district": "Huyện Đông Anh",
        "lat": 21.1372,
        "lon": 105.8441,
        "color": [245, 158, 11, 220],
        "hex": "#f59e0b",
        "feature": "Đồng bằng bãi bồi, thoáng gió sông Hồng",
        "zone_type": "Ngoại ô đồng bằng"
    },
    "st06_socson": {
        "id": "ST06_SOCSON",
        "name": "Sóc Sơn (Vùng Đồi Gò Ngoại Thành)",
        "district": "Huyện Sóc Sơn",
        "lat": 21.2847,
        "lon": 105.8542,
        "color": [6, 182, 212, 220],
        "hex": "#06b6d4",
        "feature": "Vùng bán sơn địa, ban đêm nhiệt độ hạ sâu hơn nội đô 2.5 - 3°C",
        "zone_type": "Vùng sinh thái / Đồi gò"
    }
}


# ==============================================================================
# 3. THANH ĐIỀU KHIỂN BÊN TRÁI (BỘ LỌC DÀNH CHO VẬN HÀNH)
# ==============================================================================
with st.sidebar:
    st.markdown("### 🌐 METEO-HANOI™")
    st.caption("Weather Intelligence & Operations")
    st.markdown("---")

    selected_loc_key = st.selectbox(
        "Khu vực quan tâm:",
        options=list(STATIONS_METADATA.keys()),
        format_func=lambda x: STATIONS_METADATA[x]["name"],
        index=0
    )

    horizon_choice = st.radio(
        "Khung thời gian dự báo:",
        options=[24, 48, 72],
        format_func=lambda x: f"{x} giờ tới ({x // 24} ngày)",
        index=0
    )

    st.markdown("---")
    
    with st.expander("📤 Nạp dữ liệu cảm biến mới (Tùy chọn)"):
        st.write("Hỗ trợ nạp đồng thời cả 6 tệp CSV từ mạng lưới trạm tự động:")
        uploaded_files = st.file_uploader(
            "Chọn tệp CSV dữ liệu trạm:",
            type=["csv"],
            accept_multiple_files=True
        )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; line-height:1.4;'>
      Hệ thống vận hành liên tục 24/7.<br>
      Dữ liệu cảm biến vi khí hậu tự động đồng bộ mỗi giờ.<br>
      Tầm nhìn dự báo vi khí hậu chính xác cao.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. CHẠY SUY DIỄN DỮ LIỆU THỰC TẾ (REAL INFERENCE ENGINE)
# ==============================================================================
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
        for k in list(STATIONS_METADATA.keys())[1:]:
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

forecast_results = {}
if stations_data:
    for k, df_s in stations_data.items():
        try:
            res = engine.predict(df_s, horizon=horizon, use_exogenous=True)
            forecast_results[k] = res
        except Exception:
            pass


# ==============================================================================
# 5. GIAO DIỆN SẢN PHẨM THỰC THẾ (EXECUTIVE METEOROLOGICAL PLATFORM)
# ==============================================================================
# Header cấp cao
st.markdown("""
<div class="executive-header">
  <div>
    <div class="brand-title">METEO-HANOI INTELLIGENCE™</div>
    <div class="brand-subtitle">Nền Tảng Giám Sát & Dự Báo Vi Khí Hậu Đô Thị Thời Gian Thực</div>
  </div>
  <div style="display:flex; gap:12px; align-items:center;">
    <div class="status-badge">
      <div class="status-dot"></div>
      MẠNG LƯỚI QUAN TRẮC ONLINE (6/6 TRẠM)
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not forecast_results:
    st.info("Đang đồng bộ dữ liệu quan trắc vi khí hậu...")
else:
    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 1: TỔNG QUAN TOÀN THÀNH PHỐ HÀ NỘI
    # --------------------------------------------------------------------------
    if len(forecast_results) > 1:
        current_temps = [r['metrics']['current_temp'] for r in forecast_results.values()]
        max_temps = [r['metrics']['predicted_max'] for r in forecast_results.values()]
        min_temps = [r['metrics']['predicted_min'] for r in forecast_results.values()]
        
        avg_current = round(float(np.mean(current_temps)), 1)
        highest_temp = max(max_temps)
        lowest_temp = min(min_temps)
        uhi_gap = round(max(current_temps) - min(current_temps), 1)

        # Tìm trạm nóng nhất & trạm mát nhất
        hottest_st_key = max(forecast_results.keys(), key=lambda k: forecast_results[k]['metrics']['predicted_max'])
        hottest_st_name = STATIONS_METADATA.get(hottest_st_key, {}).get("district", hottest_st_key)
        
        coldest_st_key = min(forecast_results.keys(), key=lambda k: forecast_results[k]['metrics']['predicted_min'])
        coldest_st_name = STATIONS_METADATA.get(coldest_st_key, {}).get("district", coldest_st_key)

        # 1. HỘP KHUYẾN NGHỊ VẬN HÀNH (TOMORROW.IO OPERATIONAL ADVISORY)
        st.markdown(f"""
        <div class="advisory-card">
          <div class="advisory-title">⚡ Khuyến Cáo Vận Hành Đô Thị & Sức Khỏe Cộng Đồng</div>
          <div class="advisory-content">
            Hiệu ứng <b>Đảo nhiệt đô thị (UHI)</b> đang duy trì mức chênh lệch <b>+{uhi_gap}°C</b> giữa khu vực nội đô và ngoại thành.
            Khu vực <b>{hottest_st_name}</b> dự kiến chịu áp lực nhiệt cao nhất, đạt đỉnh <b>{highest_temp}°C</b> vào khung giờ trưa (12:30 - 15:30).
            Khuyến nghị các đơn vị vận tải, logistics ngoài trời và công trường điều chỉnh ca kíp tránh giờ nắng gắt.
            Về đêm, nhiệt độ toàn thành phố hạ sâu nhất tại <b>{coldest_st_name}</b> xuống <b>{lowest_temp}°C</b>, điều kiện vi khí hậu thuận lợi.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. HỆ THỐNG 4 THẺ CHỈ SỐ VẬN HÀNH (METRICS TILES)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Nhiệt độ hiện tại</div>
              <div class="kpi-value">{avg_current}°C</div>
              <div class="kpi-subtext">Trung bình toàn địa bàn</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Đỉnh nhiệt cao nhất</div>
              <div class="kpi-value" style="color:#f87171;">{highest_temp}°C</div>
              <div class="kpi-subtext">Ghi nhận tại {hottest_st_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Đáy nhiệt ban đêm</div>
              <div class="kpi-value" style="color:#38bdf8;">{lowest_temp}°C</div>
              <div class="kpi-subtext">Hạ nhiệt tại {coldest_st_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Chênh lệch Đảo Nhiệt (UHI)</div>
              <div class="kpi-value" style="color:#fbbf24;">+{uhi_gap}°C</div>
              <div class="kpi-subtext">Nội thành vs Vùng sinh thái</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. BẢN ĐỒ VI KHÍ HẬU 3D TƯƠNG TÁC (TOMORROW.IO SPATIAL MAP VỚI PYDECK)
        st.markdown("""
        <div class="section-title">
          <span>🗺️ Bản Đồ Không Gian Nhiệt Độ Trực Quan Hà Nội</span>
        </div>
        """, unsafe_allow_html=True)

        map_data = []
        for k, res in forecast_results.items():
            meta = STATIONS_METADATA.get(k, {})
            if meta and "lat" in meta:
                cur_t = res['metrics']['current_temp']
                max_t = res['metrics']['predicted_max']
                # Tỉ lệ màu sắc theo nhiệt độ
                r = int(min(255, max(0, (cur_t - 20) * 15 + 100)))
                g = int(max(40, 200 - (cur_t - 20) * 8))
                b = int(max(20, 255 - (cur_t - 20) * 18))
                
                map_data.append({
                    "name": meta["name"],
                    "district": meta["district"],
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "temp": cur_t,
                    "max_temp": max_t,
                    "feature": meta["feature"],
                    "color": [r, g, b, 210],
                    "elevation": (cur_t - 15) * 80
                })

        map_df = pd.DataFrame(map_data)

        # Các tầng hiển thị PyDeck
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_elevation="elevation",
            elevation_scale=50,
            radius=1600,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius=3200,
            get_fill_color="color",
            pickable=True,
            opacity=0.4
        )

        text_layer = pdk.Layer(
            "TextLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_text="district",
            get_size=15,
            get_color=[240, 240, 240, 255],
            get_alignment_baseline="'bottom'",
            pickable=False
        )

        view_state = pdk.ViewState(
            latitude=21.08,
            longitude=105.83,
            zoom=9.6,
            pitch=45,
            bearing=0
        )

        deck = pdk.Deck(
            layers=[scatter_layer, column_layer, text_layer],
            initial_view_state=view_state,
            map_style=pdk.map_styles.CARTO_DARK,
            tooltip={
                "html": "<b style='color:#38bdf8;'>{name}</b><br/>"
                        "Nhiệt độ hiện tại: <b>{temp}°C</b><br/>"
                        "Đỉnh nhiệt dự báo: <b>{max_temp}°C</b><br/>"
                        "Đặc tính: <i>{feature}</i>",
                "style": {"backgroundColor": "#0f172a", "color": "#ffffff", "fontSize": "12px"}
            }
        )

        st.pydeck_chart(deck, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. THANH DẢI NHIỆT ĐỘ CÁC QUẬN (APPLE WEATHER STYLE RANGE BARS)
        st.markdown("""
        <div class="section-title">
          <span>📊 Dải Biến Thiên Nhiệt Độ Đa Vùng (Min - Hiện Tại - Max)</span>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])
        items_list = list(forecast_results.items())
        
        for idx, (st_k, res) in enumerate(items_list):
            m = res['metrics']
            meta = STATIONS_METADATA.get(st_k, {})
            name_label = meta.get("name", st_k)
            t_curr = m['current_temp']
            t_min = m['predicted_min']
            t_max = m['predicted_max']

            target_col = col_left if idx % 2 == 0 else col_right
            with target_col:
                st.markdown(f"""
                <div class="temp-bar-container">
                  <div class="station-name-col">📍 {name_label}</div>
                  <div style="font-size:0.85rem; color:#38bdf8; font-weight:700;">{t_min:.1f}°</div>
                  <div class="bar-wrapper">
                    <div class="bar-fill" style="width: 100%;"></div>
                  </div>
                  <div style="font-size:0.85rem; color:#f87171; font-weight:700;">{t_max:.1f}°</div>
                  <div style="min-width:65px; text-align:right; font-weight:800; font-size:1.05rem; color:#ffffff;">
                    {t_curr:.1f}°C
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 5. BIỂU ĐỒ METEOGRAM ĐA TRẠM CHUYÊN NGHIỆP (METEOBLUE STYLE)
        st.markdown("""
        <div class="section-title">
          <span>📈 Đồ Thị Sóng Nhiệt Theo Giờ (Multi-Station Meteogram)</span>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(14, 5.6), dpi=130)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')

        first_res = list(forecast_results.values())[0]
        timestamps = first_res['predictions']['timestamp'].dt.strftime('%H:%M (%d/%m)').values
        step_stride = max(1, len(timestamps) // 8)
        x_indices = np.arange(len(timestamps))

        # Phân vùng ngày đêm trực quan
        hours_arr = first_res['predictions']['timestamp'].dt.hour.values
        for i in range(len(hours_arr)):
            h = hours_arr[i]
            if 6 <= h <= 18:
                ax.axvspan(i - 0.5, i + 0.5, color='#fbbf24', alpha=0.06)
            else:
                ax.axvspan(i - 0.5, i + 0.5, color='#38bdf8', alpha=0.04)

        for idx, (st_k, res) in enumerate(forecast_results.items()):
            temps = res['predictions']['predicted_temperature'].values
            info = STATIONS_METADATA.get(st_k, {})
            loc_label = info.get("district", st_k)
            loc_color = info.get("hex", "#38bdf8")

            ax.plot(x_indices, temps, color=loc_color, linewidth=2.4, marker='o', markersize=3.5, label=f"{loc_label} ({temps.min():.1f}° - {temps.max():.1f}°C)")

        ax.set_xticks(x_indices[::step_stride])
        ax.set_xticklabels(timestamps[::step_stride], rotation=0, fontsize=9, color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        ax.set_ylabel("Nhiệt độ không khí (°C)", fontsize=10, labelpad=8, color='#94a3b8')
        ax.grid(True, linestyle='--', alpha=0.15, color='#94a3b8')
        
        for spine in ax.spines.values():
            spine.set_color('#334155')

        ax.legend(loc='upper right', frameon=True, facecolor='#0f172a', edgecolor='#334155', fontsize=9, ncol=2, labelcolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # 6. BẢNG SO SÁNH & XUẤT BÁO CÁO DOANH NGHIỆP (EXPORT CSV)
        st.markdown("""
        <div class="section-title">
          <span>📋 Bảng Tổng Hợp Tham Số Vi Khí Hậu Từng Phân Vùng</span>
        </div>
        """, unsafe_allow_html=True)

        summary_rows = []
        for st_k, res in forecast_results.items():
            temps = res['predictions']['predicted_temperature'].values
            info = STATIONS_METADATA.get(st_k, {})
            summary_rows.append({
                "Trạm Quan Trắc": info.get("name", st_k),
                "Phân Vùng Đô Thị": info.get("zone_type", "Đô thị"),
                "Nhiệt Độ Hiện Tại": f"{res['metrics']['current_temp']}°C",
                "Đáy Nhiệt Đêm": f"{temps.min():.1f}°C",
                "Đỉnh Nhiệt Trưa": f"{temps.max():.1f}°C",
                "Biên Độ Ngày/Đêm": f"{(temps.max() - temps.min()):.1f}°C",
                "Đặc Tính Vi Khí Hậu": info.get("feature", "Bình thường")
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # Tạo file CSV tải về
        csv_export = pd.DataFrame({"Thời gian": timestamps})
        for st_k, res in forecast_results.items():
            loc_short = STATIONS_METADATA.get(st_k, {}).get("district", st_k)
            csv_export[f"Nhiệt độ {loc_short} (°C)"] = res['predictions']['predicted_temperature'].values

        csv_buf = io.StringIO()
        csv_export.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Xuất Báo Cáo Chuỗi Thời Gian Vi Khí Hậu Hà Nội (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"meteo_hanoi_forecast_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 2: XEM CHI TIẾT 1 TRẠM DUY NHẤT
    # --------------------------------------------------------------------------
    else:
        st_k = list(forecast_results.keys())[0]
        res = forecast_results[st_k]
        pred_df = res['predictions']
        history_df = res['lookback_history']
        m = res['metrics']
        meta = STATIONS_METADATA.get(st_k, {})
        loc_name = meta.get("name", st_k)

        # 1. HỘP KHUYẾN NGHỊ VẬN HÀNH CHO TRẠM
        peak_idx = int(np.argmax(pred_df['predicted_temperature'].values))
        peak_time = pred_df['timestamp'].iloc[peak_idx].strftime('%H:%M ngày %d/%m')
        
        low_idx = int(np.argmin(pred_df['predicted_temperature'].values))
        low_time = pred_df['timestamp'].iloc[low_idx].strftime('%H:%M ngày %d/%m')

        st.markdown(f"""
        <div class="advisory-card">
          <div class="advisory-title">⚡ Cảnh Báo Vi Khí Hậu: {loc_name}</div>
          <div class="advisory-content">
            Khu vực <b>{meta.get('district', loc_name)}</b> được dự báo sẽ đạt đỉnh nhiệt <b>{m['predicted_max']}°C</b> vào lúc <b>{peak_time}</b>.
            Đêm nay và rạng sáng, nhiệt độ sẽ giảm xuống mức thấp nhất <b>{m['predicted_min']}°C</b> vào lúc <b>{low_time}</b>.
            Đặc trưng phân vùng: <i>{meta.get('feature', 'Bình thường')}</i>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. 4 THẺ CHỈ SỐ CỦA TRẠM
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Nhiệt độ hiện tại</div>
              <div class="kpi-value">{m['current_temp']}°C</div>
              <div class="kpi-subtext">Quan trắc tức thời</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Đỉnh nhiệt cao nhất</div>
              <div class="kpi-value" style="color:#f87171;">{m['predicted_max']}°C</div>
              <div class="kpi-subtext">Lúc {peak_time}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Đáy nhiệt ban đêm</div>
              <div class="kpi-value" style="color:#38bdf8;">{m['predicted_min']}°C</div>
              <div class="kpi-subtext">Lúc {low_time}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Biên độ nhiệt ngày đêm</div>
              <div class="kpi-value" style="color:#fbbf24;">{m['amplitude']}°C</div>
              <div class="kpi-subtext">Mức dao động nhiệt</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. BĂNG CHUYỀN THỜI TIẾT TỪNG GIỜ (APPLE WEATHER STYLE HOURLY RIBBON)
        st.markdown("""
        <div class="section-title">
          <span>🕒 Dòng Thời Gian Dự Báo Từng Giờ (Hourly Forecast Ribbon)</span>
        </div>
        """, unsafe_allow_html=True)

        hourly_html = ['<div class="hourly-container">']
        for i in range(min(24, len(pred_df))):
            t_val = pred_df['predicted_temperature'].iloc[i]
            ts = pred_df['timestamp'].iloc[i]
            hour_num = ts.hour
            time_str = ts.strftime('%H:00')

            # Icon theo giờ và nhiệt độ
            if 6 <= hour_num <= 17:
                icon = "☀️" if t_val >= 33 else "🌤️"
            else:
                icon = "🌙"

            hourly_html.append(f"""
            <div class="hourly-card">
              <div class="hourly-time">{time_str}</div>
              <div class="hourly-icon">{icon}</div>
              <div class="hourly-temp">{t_val:.1f}°</div>
            </div>
            """)
        hourly_html.append('</div>')
        st.markdown("".join(hourly_html), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. BIỂU ĐỒ METEOGRAM QUAN SÁT & DỰ BÁO
        st.markdown(f"""
        <div class="section-title">
          <span>📈 Chuỗi Thời Gian Quan Trắc Lịch Sử & Dự Báo Tiếp Diễn ({loc_name})</span>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(14, 5.2), dpi=130)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')

        hist_temps = history_df['temperature'].values
        pred_temps = pred_df['predicted_temperature'].values
        
        all_x = np.arange(len(hist_temps) + len(pred_temps))
        hist_x = all_x[:len(hist_temps)]
        pred_x = all_x[len(hist_temps):]

        # Vẽ đường lịch sử & tương lai
        ax.plot(hist_x, hist_temps, color='#64748b', linewidth=2.0, label='Quan trắc gần đây (96h)')
        ax.plot(pred_x, pred_temps, color='#38bdf8', linewidth=2.8, marker='o', markersize=4, label='Dự báo tiếp diễn')

        # Đánh dấu đỉnh trưa và đáy đêm
        ax.scatter([pred_x[peak_idx]], [pred_temps[peak_idx]], color='#ef4444', s=120, zorder=5, label=f'Đỉnh nhiệt ({pred_temps[peak_idx]:.1f}°C)')
        ax.scatter([pred_x[low_idx]], [pred_temps[low_idx]], color='#06b6d4', s=120, zorder=5, label=f'Đáy nhiệt ({pred_temps[low_idx]:.1f}°C)')

        ax.axvline(x=len(hist_temps) - 1, color='#f97316', linestyle='--', linewidth=1.8, label='Mốc hiện tại')
        ax.tick_params(colors='#94a3b8')
        ax.set_ylabel("Nhiệt độ (°C)", fontsize=10, labelpad=8, color='#94a3b8')
        ax.grid(True, linestyle='--', alpha=0.15, color='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

        ax.legend(loc='upper left', frameon=True, facecolor='#0f172a', edgecolor='#334155', fontsize=9, labelcolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # 5. MA TRẬN 4 THAM SỐ NGOẠI SINH DẪN DẮT (ENVIRONMENTAL SENSORS)
        st.markdown("""
        <div class="section-title">
          <span>🌿 Ma Trận Tham Số Khí Tượng Ngoại Sinh Dẫn Dắt</span>
        </div>
        """, unsafe_allow_html=True)

        # Lấy giá trị trạm mới nhất từ history_df
        last_row = history_df.iloc[-1]
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            rad_val = round(float(last_row.get('solar_radiation', 0)), 1)
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Bức Xạ Mặt Trời</div>
              <div class="kpi-value" style="color:#fbbf24; font-size:1.8rem;">{rad_val} W/m²</div>
              <div class="kpi-subtext">Cường độ nắng & UV</div>
            </div>
            """, unsafe_allow_html=True)
        with e2:
            hum_val = round(float(last_row.get('humidity', 0)), 1)
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Độ Ẩm Tương Đối</div>
              <div class="kpi-value" style="color:#38bdf8; font-size:1.8rem;">{hum_val}%</div>
              <div class="kpi-subtext">Độ ẩm không khí trạm</div>
            </div>
            """, unsafe_allow_html=True)
        with e3:
            press_val = round(float(last_row.get('surface_pressure', 0)), 1)
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Khí Áp Bề Mặt</div>
              <div class="kpi-value" style="color:#818cf8; font-size:1.8rem;">{press_val} hPa</div>
              <div class="kpi-subtext">Áp suất khí quyển</div>
            </div>
            """, unsafe_allow_html=True)
        with e4:
            wind_val = round(float(last_row.get('wind_speed_10m', 0)), 1)
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Tốc Độ Gió</div>
              <div class="kpi-value" style="color:#34d399; font-size:1.8rem;">{wind_val} m/s</div>
              <div class="kpi-subtext">Tốc độ gió tầng 10m</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 6. BẢNG DỰ BÁO CHI TIẾT & NÚT XUẤT CSV
        st.markdown("""
        <div class="section-title">
          <span>📋 Bảng Số Liệu Chi Tiết Từng Giờ</span>
        </div>
        """, unsafe_allow_html=True)

        detail_df = pd.DataFrame({
            "Thời Gian": pred_df['timestamp'].dt.strftime('%H:%M ngày %d/%m/%Y'),
            "Nhiệt Độ Dự Báo (°C)": pred_df['predicted_temperature'],
            "Xu Hướng": ["Đang tăng nhiệt" if i > 0 and pred_temps[i] > pred_temps[i-1] else "Đang hạ nhiệt" for i in range(len(pred_temps))]
        })
        st.dataframe(detail_df, use_container_width=True, height=220)

        csv_buf = io.StringIO()
        detail_df.to_csv(csv_buf, index=False)
        st.download_button(
            label=f"📥 Tải Tệp Dữ Liệu Dự Báo ({meta.get('district', st_k)})",
            data=csv_buf.getvalue(),
            file_name=f"du_bao_{st_k}_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
