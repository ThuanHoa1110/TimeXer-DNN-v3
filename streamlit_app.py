
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
    page_title="TimeXer-DINOv3 Weather AI | Multi-Station Serving",
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
    st.markdown("---")
    
    # 1. Chọn Horizon
    horizon = st.radio(
        "Khoảng thời gian dự báo tương lai (Horizon):",
        options=[24, 48, 72],
        format_func=lambda x: f"{x} bước ({x*5//60} giờ)",
        index=0
    )

    # 2. Biến ngoại sinh
    use_exo = st.toggle("Bật Cross-Attention Ngoại sinh (Nắng, Gió, Ẩm, Áp)", value=True)

    # 3. Lựa chọn Mô hình
    model_opt = st.selectbox(
        "Mô hình phục vụ:",
        options=["TimeXer-DINOv3 (Proposed Architecture)", "TimeXer Base (NeurIPS 2024)", "Univariate Baseline (No Exogenous)"],
        index=0
    )


# ==============================================================================
# 4. GIAO DIỆN CHÍNH: NẠP DỮ LIỆU ĐẦU VÀO ĐA TRẠM THỰC TẾ
# ==============================================================================
st.title("🌦️ Bảng Điều Khiển Dự Báo Khí Tượng Đa Trạm (TimeXer-DINOv3)")

# Định nghĩa các trạm mẫu và màu sắc biểu đồ
STATION_INFO = {
    "st01_hoankiem": {"name": "ST01 - Hoàn Kiếm (Lõi đô thị UHI +1.2°C)", "color": "#ef4444", "short": "Hoàn Kiếm"},
    "st02_badinh":   {"name": "ST02 - Ba Đình (Điều hòa Hồ Tây +0.4°C)", "color": "#0284c7", "short": "Ba Đình"},
    "st03_caugiay":  {"name": "ST03 - Cầu Giấy (Mật độ giao thông cao)", "color": "#8b5cf6", "short": "Cầu Giấy"},
    "st04_hadong":   {"name": "ST04 - Hà Đông (Trạm tham chiếu chuẩn KTTV)", "color": "#10b981", "short": "Hà Đông"},
    "st05_donganh":  {"name": "ST05 - Đông Anh (Ven đô Bắc sông Hồng)", "color": "#f59e0b", "short": "Đông Anh"},
    "st06_socson":   {"name": "ST06 - Sóc Sơn (Gò đồi bán sơn địa, đêm hạ sâu)", "color": "#06b6d4", "short": "Sóc Sơn"}
}

input_tab1, input_tab2 = st.tabs(["🛰️ Chọn Lát Cắt Mạng Lưới Từ Kho 6 Trạm Hà Nội", "📤 Tải Lên Tệp CSV (Hỗ Trợ Cả 6 Trạm Đồng Thời)"])

# Trạng thái dữ liệu đa trạm: {station_name: dataframe}
stations_dict = {}
is_multi_mode = False
ground_truth_dict = {}

with input_tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st_select_options = ["all_network"] + list(STATION_INFO.keys())
        st_labels = {
            "all_network": "🌐 TOÀN BỘ MẠNG LƯỚI 6 TRẠM ĐỒNG THỜI (Hà Nội Multi-Station Network)",
            **{k: v["name"] for k, v in STATION_INFO.items()}
        }
        selected_st_key = st.selectbox(
            "Chọn phạm vi dự báo quan trắc:",
            options=st_select_options,
            format_func=lambda x: st_labels[x],
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

    # Nạp dữ liệu từ kho trạm mẫu
    lookback_len = 96
    total_need = lookback_len + horizon

    if selected_st_key == "all_network":
        is_multi_mode = True
        for k in STATION_INFO.keys():
            st_path = os.path.join(BASE_DIR, "data", "stations_sample", f"{k}.csv")
            if os.path.exists(st_path):
                df_st = pd.read_csv(st_path)
                df_st['timestamp'] = pd.to_datetime(df_st['timestamp'])
                
                if "nắng nóng" in event_choice.lower():
                    s_pos = min(len(df_st) - total_need - 1, 400)
                elif "rét đậm" in event_choice.lower():
                    s_pos = min(len(df_st) - total_need - 1, 800)
                elif "chuyển mùa" in event_choice.lower():
                    s_pos = min(len(df_st) - total_need - 1, 1200)
                else:
                    s_pos = max(0, len(df_st) - total_need - 20)

                stations_dict[k] = df_st.iloc[s_pos : s_pos + lookback_len].copy()
                ground_truth_dict[k] = df_st.iloc[s_pos + lookback_len : s_pos + total_need].copy()
    else:
        is_multi_mode = False
        st_path = os.path.join(BASE_DIR, "data", "stations_sample", f"{selected_st_key}.csv")
        if os.path.exists(st_path):
            df_st = pd.read_csv(st_path)
            df_st['timestamp'] = pd.to_datetime(df_st['timestamp'])
            s_pos = max(0, len(df_st) - total_need - 20)
            stations_dict[selected_st_key] = df_st.iloc[s_pos : s_pos + lookback_len].copy()
            ground_truth_dict[selected_st_key] = df_st.iloc[s_pos + lookback_len : s_pos + total_need].copy()

with input_tab2:
    uploaded_files = st.file_uploader(
        "Kéo thả tệp CSV:",
        type=["csv"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.success(f"Đã nạp {len(uploaded_files)} tệp CSV.")
        stations_dict = {}
        ground_truth_dict = {}
        for f in uploaded_files:
            try:
                raw_df = pd.read_csv(f)
                stations_dict[f.name.replace(".csv", "")] = raw_df
            except Exception as e:
                st.error(f"Lỗi khi đọc file {f.name}: {e}")
        
        is_multi_mode = len(stations_dict) > 1

# Hiển thị tóm tắt dữ liệu đầu vào đã nạp
if stations_dict:
    with st.expander(f"🔍 Danh sách {len(stations_dict)} trạm đã sẵn sàng nạp vào mô hình (Xem trước dữ liệu)", expanded=False):
        for st_id, df_data in stations_dict.items():
            st.markdown(f"**Trạm: {st_id}** — Số dòng: `{len(df_data)}`, Cột: `{list(df_data.columns)[:6]}`")
            st.dataframe(df_data.tail(3), use_container_width=True)

# Nút thực thi suy luận chính
st.markdown("---")
btn_col1, btn_col2 = st.columns([1, 2])
with btn_col1:
    run_btn = st.button("⚡ CHẠY SUY LUẬN MÔ HÌNH PYTORCH", type="primary", use_container_width=True)


# ==============================================================================
# 5. THỰC THI SUY LUẬN & TRỰC QUAN HÓA KẾT QUẢ
# ==============================================================================
if run_btn:
    if not stations_dict:
        st.error("Chưa có dữ liệu trạm quan trắc nào được nạp!")
    else:
        results = {}
        total_start = time.perf_counter()

        with st.spinner(f"Đang chạy suy luận PyTorch song song trên {len(stations_dict)} trạm..."):
            for st_id, df_st in stations_dict.items():
                try:
                    res = engine.predict(df_input=df_st, horizon=horizon, use_exogenous=use_exo)
                    results[st_id] = res
                except Exception as e:
                    st.error(f"Lỗi khi suy luận trạm {st_id}: {e}")

        total_latency = (time.perf_counter() - total_start) * 1000.0

        if results:
            st.success(f"✅ Hoàn thành suy luận thành công cho **{len(results)} trạm quan trắc** trong **{total_latency:.1f} ms**!")

            # ------------------------------------------------------------------
            # TRƯỜNG HỢP 1: DỰ BÁO ĐỒNG THỜI MẠNG LƯỚI ĐA TRẠM (MULTI-STATION)
            # ------------------------------------------------------------------
            if len(results) > 1:
                # Tính toán chênh lệch đảo nhiệt đô thị (Urban Heat Island - UHI)
                all_current_temps = [r['metrics']['current_temp'] for r in results.values()]
                all_predicted_maxs = [r['metrics']['predicted_max'] for r in results.values()]
                uhi_gap = max(all_current_temps) - min(all_current_temps)
                highest_temp = max(all_predicted_maxs)
                avg_temp = np.mean(all_current_temps)

                # 4 Thẻ KPI Mạng lưới
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.markdown(f"""
                    <div class="kpi-box">
                      <div class="kpi-title">Mạng lưới quan trắc</div>
                      <div class="kpi-value">{len(results)} Trạm</div>
                      <div class="kpi-sub">Đồng bộ 100% PyTorch</div>
                    </div>
                    """, unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""
                    <div class="kpi-box">
                      <div class="kpi-title">Chênh lệch Đảo nhiệt (UHI Gap)</div>
                      <div class="kpi-value">+{uhi_gap:.1f}°C</div>
                      <div class="kpi-sub">Lõi đô thị vs Ven đô/Gò đồi</div>
                    </div>
                    """, unsafe_allow_html=True)
                with k3:
                    st.markdown(f"""
                    <div class="kpi-box">
                      <div class="kpi-title">Đỉnh nhiệt cao nhất mạng lưới</div>
                      <div class="kpi-value">{highest_temp:.1f}°C</div>
                      <div class="kpi-sub">Khoảng dự báo {horizon} bước</div>
                    </div>
                    """, unsafe_allow_html=True)
                with k4:
                    st.markdown(f"""
                    <div class="kpi-box">
                      <div class="kpi-title">Tổng thời gian suy luận</div>
                      <div class="kpi-value">{total_latency:.1f} ms</div>
                      <div class="kpi-sub">~{total_latency/len(results):.1f} ms / trạm</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 🌐 Biểu Đồ So Sánh Dự Báo Mạng Lưới 6 Trạm Khí Tượng Hà Nội")

                # Vẽ biểu đồ đa trạm
                fig, ax = plt.subplots(figsize=(14, 5.8), dpi=130)
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#fdfdfd')

                palette = ["#ef4444", "#0284c7", "#8b5cf6", "#10b981", "#f59e0b", "#06b6d4", "#ec4899", "#84cc16"]
                all_preds_series = {}

                for i, (st_id, res) in enumerate(results.items()):
                    pred_df = res['predictions']
                    st_color = STATION_INFO.get(st_id, {}).get("color", palette[i % len(palette)])
                    st_name = STATION_INFO.get(st_id, {}).get("short", st_id)

                    pred_vals = pred_df['predicted_temperature'].values
                    all_preds_series[st_name] = pred_vals
                    steps_x = np.arange(1, len(pred_vals) + 1)

                    ax.plot(steps_x, pred_vals, color=st_color, linewidth=2.4, marker='o', markersize=3.5, label=f"{st_name} ({pred_vals.min():.1f}° - {pred_vals.max():.1f}°C)")

                ax.set_title(f"Dự báo Nhiệt độ Toàn Mạng Lưới {len(results)} Trạm • Horizon {horizon} Bước (TimeXer-DINOv3)", fontsize=12, fontweight='bold', pad=12)
                ax.set_xlabel("Bước thời gian dự báo tương lai (+5 phút/bước)", fontsize=10, labelpad=8)
                ax.set_ylabel("Nhiệt độ không khí (°C)", fontsize=10, labelpad=8)
                ax.grid(True, linestyle='--', alpha=0.35)
                ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=8.5, ncol=2)
                plt.tight_layout()
                st.pyplot(fig)

                # Bảng đối chiếu so sánh toàn diện 6 trạm
                col_tab1, col_tab2 = st.columns([1.2, 0.8])
                with col_tab1:
                    st.markdown("#### 📊 Bảng Đối Chiếu So Sánh Đa Trạm Mạng Lưới")
                    comparison_rows = []
                    for st_id, res in results.items():
                        m = res['metrics']
                        attr = res['exogenous_attribution']
                        top_exo = max(attr.items(), key=lambda x: x[1])
                        comparison_rows.append({
                            "Mã Trạm": st_id,
                            "Tên Trạm": STATION_INFO.get(st_id, {}).get("short", st_id),
                            "Hiện Tại (°C)": m['current_temp'],
                            "Min (°C)": m['predicted_min'],
                            "Max (°C)": m['predicted_max'],
                            "Biên Độ ΔT": m['amplitude'],
                            "Yếu Tố Chi Phối Mạnh Nhất": f"{top_exo[0]} ({top_exo[1]}%)"
                        })
                    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

                with col_tab2:
                    st.markdown("#### 📥 Xuất Báo Cáo Dự Báo Tổng Hợp 6 Trạm")
                    
                    # Ghép DataFrame tổng hợp
                    first_st = list(results.keys())[0]
                    combined_df = pd.DataFrame({"timestamp": results[first_st]['predictions']['timestamp']})
                    for st_id, res in results.items():
                        st_name = STATION_INFO.get(st_id, {}).get("short", st_id)
                        combined_df[f"{st_name}_temp"] = res['predictions']['predicted_temperature'].values

                    st.dataframe(combined_df.head(6), use_container_width=True)

                    csv_all_buffer = io.StringIO()
                    combined_df.to_csv(csv_all_buffer, index=False)
                    st.download_button(
                        label="📥 TẢI TOÀN BỘ FILE CSV 6 TRẠM (COMBINED CSV)",
                        data=csv_all_buffer.getvalue(),
                        file_name=f"timexer_dinov3_6stations_forecast_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # Tab chi tiết từng trạm
                st.markdown("---")
                st.markdown("### 🔍 Phân Tích Chuyên Sâu Từng Trạm Cụ Thể (Station Deep-Dive)")
                deep_dive_st = st.selectbox(
                    "Chọn trạm để xem đồ thị chi tiết kèm điểm uốn và trọng số Explainable AI:",
                    options=list(results.keys()),
                    format_func=lambda x: STATION_INFO.get(x, {}).get("name", x)
                )

                st_res = results[deep_dive_st]
                m_single = st_res['metrics']
                pred_single = st_res['predictions']
                hist_single = st_res['lookback_history']
                attr_single = st_res['exogenous_attribution']

                c_left, c_right = st.columns([1, 1])
                with c_left:
                    st.markdown(f"**Trọng số đóng góp ngoại sinh tại {deep_dive_st}:**")
                    attr_df = pd.DataFrame({
                        "Yếu tố ngoại sinh": ["Bức xạ mặt trời", "Vận tốc gió 10m", "Độ ẩm tương đối", "Khí áp bề mặt"],
                        "Trọng số ảnh hưởng (%)": [attr_single['solar_radiation'], attr_single['wind_speed'], attr_single['humidity'], attr_single['pressure']]
                    })
                    st.dataframe(attr_df, use_container_width=True)

                with c_right:
                    st.markdown(f"**Bảng số liệu dự báo chi tiết ({deep_dive_st}):**")
                    st.dataframe(pred_single, use_container_width=True, height=160)

            # ------------------------------------------------------------------
            # TRƯỜNG HỢP 2: DỰ BÁO CHO 1 TRẠM DUY NHẤT
            # ------------------------------------------------------------------
            else:
                st_id = list(results.keys())[0]
                res = results[st_id]
                m = res['metrics']
                pred_df = res['predictions']
                history_df = res['lookback_history']
                attr = res['exogenous_attribution']

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.markdown(f"""
                    <div class="kpi-box">
                      <div class="kpi-title">Nhiệt độ hiện tại</div>
                      <div class="kpi-value">{m['current_temp']}°C</div>
                      <div class="kpi-sub">Trạm: {st_id}</div>
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
                    diffs = np.diff(pred_df['predicted_temperature'].values)
                    num_inflections = int(np.sum(diffs[:-1] * diffs[1:] < 0))
                    st.markdown(f"""
                    <div class="kpi-box">
                      <div class="kpi-title">Điểm uốn bắt được</div>
                      <div class="kpi-value">{num_inflections} điểm</div>
                      <div class="kpi-sub">🟢 Điểm uốn đảo chiều</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Biểu đồ chuỗi thời gian đơn trạm
                fig, ax = plt.subplots(figsize=(14, 5.2), dpi=130)
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#fdfdfd')

                hist_x = np.arange(len(history_df))
                pred_x = np.arange(len(history_df), len(history_df) + len(pred_df))

                ax.plot(hist_x, history_df['temperature'].values, color='#0284c7', linewidth=2.2, label='Quá khứ quan sát thực tế (Look-back L=96)')
                pred_vals = pred_df['predicted_temperature'].values
                ax.plot(pred_x, pred_vals, color='#10b981', linewidth=2.6, marker='o', markersize=4, label='Dự báo TimeXer-DINOv3 (PyTorch Tensor Forecast)')

                if st_id in ground_truth_dict and ground_truth_dict[st_id] is not None:
                    gt_vals = ground_truth_dict[st_id]['temperature'].values[:len(pred_df)]
                    ax.plot(pred_x, gt_vals, color='#334155', linestyle='--', linewidth=1.8, label='Giá trị thực tế kiểm chứng (Ground Truth)')

                ax.axvline(x=len(history_df) - 1, color='#ef4444', linestyle=':', linewidth=1.8, label='Mốc phân tách T_0')
                ax.set_title(f"Kết quả Dự báo TimeXer-DINOv3 • Trạm: {st_id}", fontsize=12, fontweight='bold', pad=12)
                ax.set_xlabel("Bước thời gian", fontsize=10, labelpad=8)
                ax.set_ylabel("Nhiệt độ (°C)", fontsize=10, labelpad=8)
                ax.grid(True, linestyle='--', alpha=0.35)
                ax.legend(loc='upper left', frameon=True, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

                # Tải file CSV
                csv_buffer = io.StringIO()
                pred_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📥 Tải tệp CSV kết quả dự báo ({st_id})",
                    data=csv_buffer.getvalue(),
                    file_name=f"timexer_dinov3_prediction_{st_id}_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
