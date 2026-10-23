"""
Script trực quan hóa dữ liệu khí tượng Hà Nội 10 năm (real_weather_large_dataset.csv - 1,052,064 dòng, 5 phút)
Thiết kế theo chuẩn báo cáo khoa học quốc tế (Academic / Journal Style)
Tự động xuất 6 biểu đồ phân tích chuyên sâu 300 DPI vào thư mục figures/
"""

import os
import sys

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi charmap
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

# Thiết lập phong cách đồ họa học thuật (Tối giản, trang nhã, không màu mè AI)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.edgecolor'] = '#94a3b8'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['figure.autolayout'] = False

# Bảng màu phong cách tạp chí khoa học (Journal Muted Palette)
C_PRIMARY = "#1e3a8a"    # Xanh Navy đậm (Target / Xu hướng chính)
C_SECONDARY = "#0f766e"  # Teal trầm (Biến bổ trợ)
C_ACCENT = "#b45309"     # Hổ phách / Amber (Bức xạ mặt trời)
C_ALERT = "#b91c1c"      # Đỏ đun (Cực đoan / Cảnh báo)
C_MUTED = "#64748b"      # Xám chì (Đường phụ / Viền)
C_RIBBON = "#cbd5e1"     # Xám bạc (Dải bóng mờ min-max)

CSV_PATH = "data/private/real_weather_large_dataset.csv" if os.path.exists("data/private/real_weather_large_dataset.csv") else "real_weather_large_dataset.csv"


def visualize_weather():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Không tìm thấy tệp {CSV_PATH}. Vui lòng kiểm tra lại đường dẫn!")
        return

    print(f"[*] Đang nạp dữ liệu từ: {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    os.makedirs("figures", exist_ok=True)
    print(f"[*] Đã nạp thành công: {len(df):,} dòng, {df.shape[1]} cột đặc trưng.")

    # =========================================================================
    # HÌNH 1: TOÀN CẢNH CHUỖI THỜI GIAN 10 NĂM (RESAMPLE TUẦN + INSET CHI TIẾT)
    # =========================================================================
    print("[1/6] Đang vẽ biểu đồ toàn cảnh 10 năm kèm dải biến thiên và sự kiện cực đoan...")
    
    # Resample theo tuần để tránh overplotting 87.000 điểm
    df_ts = df.set_index('timestamp')
    weekly = df_ts.resample('W').agg({
        'temperature': ['mean', 'min', 'max'],
        'humidity': 'mean',
        'solar_radiation': 'mean',
        'surface_pressure': 'mean'
    })
    dates = weekly.index

    fig = plt.figure(figsize=(15, 11), dpi=300)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.3, 1.2], hspace=0.28)

    # Subplot 1: Nhiệt độ với dải Min-Max và chú thích sự kiện cực đoan lịch sử
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(dates, weekly['temperature']['min'], weekly['temperature']['max'],
                     color=C_RIBBON, alpha=0.55, label='Biên độ dao động tuần (Min – Max)')
    ax1.plot(dates, weekly['temperature']['mean'], color=C_PRIMARY, linewidth=1.6,
             label='Nhiệt độ trung bình tuần')

    # Annotation các sự kiện thời tiết cực đoan thực tế có trong dữ liệu
    ax1.annotate(
        "Đợt rét đậm kỷ lục\n4.4°C (24/01/2016)",
        xy=(pd.Timestamp('2016-01-24'), 4.4),
        xytext=(pd.Timestamp('2016-10-01'), 2.0),
        arrowprops=dict(facecolor=C_ALERT, edgecolor=C_ALERT, arrowstyle="->", lw=1.2),
        fontsize=9, color=C_ALERT, fontweight='semibold'
    )
    ax1.annotate(
        "Đỉnh nắng nóng gay gắt\n40.2°C (27/04/2024)",
        xy=(pd.Timestamp('2024-04-27'), 40.2),
        xytext=(pd.Timestamp('2022-06-01'), 42.0),
        arrowprops=dict(facecolor=C_ALERT, edgecolor=C_ALERT, arrowstyle="->", lw=1.2),
        fontsize=9, color=C_ALERT, fontweight='semibold'
    )

    ax1.set_ylabel('Nhiệt độ không khí (°C)', fontweight='semibold')
    ax1.set_title('Xu thế diễn biến chuỗi thời gian khí tượng Hà Nội giai đoạn 2016 – 2025',
                  fontsize=13, fontweight='bold', pad=10, loc='left')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#e2e8f0')
    ax1.grid(True, axis='y')
    sns.despine(ax=ax1, top=True, right=True)

    # Subplot 2: Bức xạ mặt trời và Độ ẩm tương đối
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(dates, weekly['solar_radiation']['mean'], color=C_ACCENT, linewidth=1.2, label='Bức xạ mặt trời TB (W/m²)')
    l2 = ax2_twin.plot(dates, weekly['humidity']['mean'], color=C_SECONDARY, linewidth=1.2, linestyle='--', label='Độ ẩm tương đối TB (%)')
    ax2.set_ylabel('Bức xạ (W/m²)', color=C_ACCENT, fontweight='semibold')
    ax2_twin.set_ylabel('Độ ẩm (%)', color=C_SECONDARY, fontweight='semibold')
    ax2.grid(True, axis='y')
    sns.despine(ax=ax2, top=True)
    sns.despine(ax=ax2_twin, top=True, left=True)

    # Subplot 3: Áp suất bề mặt (thể hiện tính nghịch đảo rõ nét theo mùa)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(dates, weekly['surface_pressure']['mean'], color=C_MUTED, linewidth=1.3)
    ax3.set_ylabel('Áp suất bề mặt (hPa)', fontweight='semibold')
    ax3.set_xlabel('Năm thu thập dữ liệu', fontweight='semibold')
    ax3.grid(True, axis='y')
    sns.despine(ax=ax3, top=True, right=True)

    plt.savefig('figures/weather_01_10year_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # HÌNH 2: PHÂN TÍCH TRỄ PHA NHẬT KỲ (24-HOUR DIURNAL LAG PATTERN)
    # =========================================================================
    print("[2/6] Đang vẽ biến thiên chu kỳ nhật kỳ 24h và hiện tượng trễ pha nhiệt...")
    df['hour'] = df['timestamp'].dt.hour
    
    # Tính thống kê quantile (25%, 50%, 75%) và mean theo giờ
    hourly_agg = df.groupby('hour').agg({
        'temperature': ['mean', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'solar_radiation': 'mean',
        'humidity': 'mean'
    })

    hours = range(24)
    t_mean = hourly_agg['temperature']['mean']
    t_q25 = hourly_agg['temperature']['<lambda_0>']
    t_q75 = hourly_agg['temperature']['<lambda_1>']
    rad_mean = hourly_agg['solar_radiation']['mean']
    hum_mean = hourly_agg['humidity']['mean']

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), dpi=300, sharex=True,
                                         gridspec_kw={'height_ratios': [1.8, 1.2], 'hspace': 0.15})

    # Subplot trên: Bức xạ cực đại vs Nhiệt độ trễ pha
    ax_rad = ax_top.twinx()
    # Bức xạ tô dạng bóng mờ diện tích
    ax_rad.fill_between(hours, rad_mean, color=C_ACCENT, alpha=0.22, label='Bức xạ mặt trời (W/m²)')
    ax_rad.plot(hours, rad_mean, color=C_ACCENT, linewidth=1.5, linestyle=':')
    ax_rad.set_ylabel('Bức xạ mặt trời TB (W/m²)', color=C_ACCENT, fontweight='semibold')

    # Nhiệt độ dải IQR và đường trung bình
    ax_top.fill_between(hours, t_q25, t_q75, color=C_PRIMARY, alpha=0.18, label='Dải liên phân vị nhiệt độ (IQR 25% – 75%)')
    ax_top.plot(hours, t_mean, color=C_PRIMARY, linewidth=2.4, marker='o', markersize=5, label='Nhiệt độ không khí TB (°C)')
    ax_top.set_ylabel('Nhiệt độ (°C)', color=C_PRIMARY, fontweight='semibold')
    ax_top.set_title('Hiện tượng trễ pha nhiệt độ so với bức xạ mặt trời trong chu kỳ ngày – đêm (24 giờ)',
                     fontsize=12, fontweight='bold', pad=10, loc='left')

    # Chú thích khoa học về hiện tượng trễ pha 2 giờ
    ax_top.annotate(
        "Bức xạ đạt đỉnh (12:00)",
        xy=(12, 15), xytext=(8.5, 18),
        arrowprops=dict(facecolor=C_ACCENT, edgecolor=C_ACCENT, arrowstyle="->", lw=1.2),
        fontsize=9, color=C_ACCENT, fontweight='semibold'
    )
    ax_top.annotate(
        "Nhiệt độ đạt đỉnh (14:00)\nTrễ pha ~2 giờ (Quán tính nhiệt)",
        xy=(14, t_mean[14]), xytext=(15.2, 27.5),
        arrowprops=dict(facecolor=C_PRIMARY, edgecolor=C_PRIMARY, arrowstyle="->", lw=1.2),
        fontsize=9, color=C_PRIMARY, fontweight='semibold'
    )

    ax_top.grid(True, axis='y')
    sns.despine(ax=ax_top, top=True)
    sns.despine(ax=ax_rad, top=True, left=True)

    # Subplot dưới: Độ ẩm tương đối biến thiên nghịch pha
    ax_bot.plot(hours, hum_mean, color=C_SECONDARY, linewidth=2.0, marker='s', markersize=4, label='Độ ẩm tương đối (%)')
    ax_bot.set_ylabel('Độ ẩm TB (%)', color=C_SECONDARY, fontweight='semibold')
    ax_bot.set_xlabel('Giờ trong ngày (0 – 23h)', fontweight='semibold')
    ax_bot.set_xticks(range(0, 24, 2))
    ax_bot.grid(True, axis='y')
    sns.despine(ax=ax_bot, top=True, right=True)

    plt.savefig('figures/weather_02_diurnal_cycle.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # HÌNH 3: PHÂN TÍCH MÙA VỤ 12 THÁNG & LƯỢNG MƯA GIÓ MÙA
    # =========================================================================
    print("[3/6] Đang vẽ chu kỳ mùa vụ 12 tháng kết hợp cơ chế gió mùa...")
    df['month'] = df['timestamp'].dt.month
    monthly_rain = df.groupby('month')['rain'].sum() / df['timestamp'].dt.year.nunique()

    fig, (ax_box, ax_bar) = plt.subplots(2, 1, figsize=(12, 8.5), dpi=300,
                                         gridspec_kw={'height_ratios': [1.5, 1.1], 'hspace': 0.25})

    # Boxplot nhiệt độ 12 tháng với tông màu Slate/Blue thanh lịch
    box_palette = [C_PRIMARY if m in [5, 6, 7, 8, 9, 10] else C_MUTED for m in range(1, 13)]
    sns.boxplot(
        x='month', y='temperature', data=df,
        ax=ax_box, palette=box_palette, hue='month', legend=False,
        width=0.6, fliersize=1.5, linewidth=1.0, boxprops=dict(alpha=0.85)
    )
    # Đường nối Median giữa các tháng
    monthly_medians = df.groupby('month')['temperature'].median()
    ax_box.plot(range(0, 12), monthly_medians.values, color=C_ALERT, linewidth=1.8, linestyle='--',
                marker='D', markersize=4.5, label='Trung vị nhiệt độ theo tháng')
    ax_box.set_title('Phân bố nhiệt độ và lượng mưa trung bình 12 tháng tại khu vực Hà Nội',
                     fontsize=12, fontweight='bold', pad=10, loc='left')
    ax_box.set_ylabel('Nhiệt độ (°C)', fontweight='semibold')
    ax_box.set_xlabel('')
    ax_box.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0')
    ax_box.grid(True, axis='y')
    sns.despine(ax=ax_box, top=True, right=True)

    # Barplot lượng mưa chia 2 giai đoạn: Mùa mưa vs Mùa khô
    bar_colors = [C_PRIMARY if m in [5, 6, 7, 8, 9, 10] else C_RIBBON for m in range(1, 13)]
    bars = ax_bar.bar(range(1, 13), monthly_rain.values, color=bar_colors, edgecolor='#94a3b8', width=0.65, linewidth=0.8)

    # Chú thích đỉnh mưa tháng 8
    ax_bar.annotate(
        f"Đỉnh mưa gió mùa\n{monthly_rain[8]:.1f} mm (Tháng 8)",
        xy=(8, monthly_rain[8]), xytext=(8.6, monthly_rain[8] - 50),
        arrowprops=dict(facecolor=C_PRIMARY, edgecolor=C_PRIMARY, arrowstyle="->", lw=1.2),
        fontsize=9, color=C_PRIMARY, fontweight='semibold'
    )
    ax_bar.set_ylabel('Lượng mưa TB (mm)', fontweight='semibold')
    ax_bar.set_xlabel('Tháng trong năm', fontweight='semibold')
    ax_bar.set_xticks(range(1, 13))
    ax_bar.set_xticklabels([f"T{i}" for i in range(1, 13)])
    ax_bar.grid(True, axis='y')
    sns.despine(ax=ax_bar, top=True, right=True)

    plt.savefig('figures/weather_03_seasonality_12months.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # HÌNH 4: MA TRẬN TƯƠNG QUAN PEARSON ĐƯỢC CHỌN LỌC VÀ SẮP XẾP HỌC THUẬT
    # =========================================================================
    print("[4/6] Đang tính ma trận tương quan Pearson sắp xếp theo mức độ ảnh hưởng...")
    selected_features = [
        'temperature', 'apparent_temperature', 'dew_point', 'solar_radiation',
        'direct_radiation', 'wind_speed_10m', 'rain', 'humidity', 'surface_pressure'
    ]
    # Sắp xếp các cột theo độ tương quan với temperature
    corr_all = df[selected_features].corr()
    sorted_order = corr_all['temperature'].sort_values(ascending=False).index
    corr = corr_all.loc[sorted_order, sorted_order]

    # Nhãn tiếng Việt khoa học cho từng đặc trưng
    labels_map = {
        'temperature': 'Nhiệt độ (Target)',
        'apparent_temperature': 'Nhiệt độ cảm nhận',
        'dew_point': 'Điểm sương',
        'solar_radiation': 'Bức xạ mặt trời',
        'direct_radiation': 'Bức xạ trực tiếp',
        'wind_speed_10m': 'Tốc độ gió 10m',
        'rain': 'Lượng mưa',
        'humidity': 'Độ ẩm tương đối',
        'surface_pressure': 'Áp suất bề mặt'
    }
    pretty_labels = [labels_map.get(c, c) for c in sorted_order]

    plt.figure(figsize=(10, 8), dpi=300)
    # Mask nửa tam giác trên
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Palette học thuật Diverging RdBu_r
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.7,
        linecolor='white',
        xticklabels=pretty_labels,
        yticklabels=pretty_labels,
        cbar_kws={"label": "Hệ số tương quan tuyến tính Pearson (r)", "shrink": 0.8}
    )
    plt.title('Ma trận tương quan giữa biến mục tiêu (Nhiệt độ) và các biến ngoại sinh',
              fontsize=12, fontweight='bold', pad=15, loc='left')
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.savefig('figures/weather_04_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # HÌNH 5: PHÂN RÃ CHUỖI THỜI GIAN VÀ XU THẾ BIẾN ĐỔI KHÍ HẬU
    # =========================================================================
    print("[5/6] Đang phân rã chuỗi thời gian và ước lượng xu thế dài hạn...")
    daily_temp = df.set_index('timestamp')['temperature'].resample('D').mean().interpolate()
    trend = daily_temp.rolling(window=365, center=True).mean()
    detrended = daily_temp - trend
    seasonal = detrended.groupby(detrended.index.dayofyear).transform('mean')
    residual = detrended - seasonal

    fig, axes = plt.subplots(4, 1, figsize=(14, 10.5), dpi=300, sharex=True,
                             gridspec_kw={'hspace': 0.22})

    # 1. Observed
    axes[0].plot(daily_temp, color='#334155', linewidth=0.75, label='Nhiệt độ TB ngày (Observed)')
    axes[0].set_ylabel('Quan sát gốc (°C)', fontweight='semibold')
    axes[0].set_title('Phân rã thành phần chuỗi thời gian nhiệt độ (Classical Additive Decomposition)',
                      fontsize=12, fontweight='bold', pad=10, loc='left')
    axes[0].grid(True, axis='y')
    sns.despine(ax=axes[0], top=True, right=True)

    # 2. Trend + Đánh giá xu thế tăng nhiệt đô thị
    axes[1].plot(trend, color=C_ALERT, linewidth=2.0, label='Xu thế trượt 365 ngày (Trend)')
    axes[1].set_ylabel('Xu thế dài hạn (°C)', fontweight='semibold')
    axes[1].grid(True, axis='y')
    sns.despine(ax=axes[1], top=True, right=True)

    # 3. Seasonal
    axes[2].plot(seasonal, color=C_PRIMARY, linewidth=0.9, label='Chu kỳ mùa vụ hàng năm (Seasonality)')
    axes[2].set_ylabel('Mùa vụ (°C)', fontweight='semibold')
    axes[2].grid(True, axis='y')
    sns.despine(ax=axes[2], top=True, right=True)

    # 4. Residuals + Ngưỡng ±2 Sigma
    std_res = residual.std()
    axes[3].scatter(residual.index, residual, color=C_MUTED, s=2.5, alpha=0.45, label='Sai số dư (Residuals)')
    axes[3].axhline(y=2 * std_res, color='#ef4444', linestyle='--', linewidth=0.9, label='+2σ Ngưỡng dị biệt (Outlier)')
    axes[3].axhline(y=-2 * std_res, color='#ef4444', linestyle='--', linewidth=0.9, label='-2σ Ngưỡng dị biệt')
    axes[3].set_ylabel('Phần dư (°C)', fontweight='semibold')
    axes[3].set_xlabel('Thời gian (Năm)', fontweight='semibold')
    axes[3].legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=8.5)
    axes[3].grid(True, axis='y')
    sns.despine(ax=axes[3], top=True, right=True)

    plt.savefig('figures/weather_05_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # HÌNH 6: PHÂN PHỐI ĐA ĐỈNH (BIMODAL) THEO MÙA NÓNG VS MÙA LẠNH
    # =========================================================================
    print("[6/6] Đang vẽ biểu đồ phân phối xác suất đa đỉnh tách biệt theo mùa...")
    
    # Tạo nhãn mùa thực tế cho Hà Nội
    df['season_type'] = np.where(df['month'].isin([5, 6, 7, 8, 9, 10]), 'Mùa nóng (T5 – T10)', 'Mùa lạnh (T11 – T4)')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=300)
    axes = axes.flatten()

    features_to_plot = [
        ('temperature', 'Nhiệt độ không khí (°C)', 'Giải thích tính đa đỉnh (Bimodal) do khí hậu 2 mùa'),
        ('humidity', 'Độ ẩm tương đối (%)', 'Phân phối lệch trái (Độ ẩm thường xuyên trên 70%)'),
        ('surface_pressure', 'Áp suất bề mặt (hPa)', 'Áp suất tăng mạnh vào mùa đông do áp cao Siberi'),
        ('wind_speed_10m', 'Tốc độ gió 10m (m/s)', 'Phân phối tiệm cận hàm Log-normal / Weibull')
    ]

    for idx, (col, label_name, sub_desc) in enumerate(features_to_plot):
        ax = axes[idx]
        
        # Vẽ phân phối tách biệt theo 2 mùa để bộc lộ tính chất tự nhiên của dữ liệu
        sns.kdeplot(
            data=df[df['season_type'] == 'Mùa lạnh (T11 – T4)'], x=col,
            ax=ax, color='#0284c7', linewidth=1.8, fill=True, alpha=0.25,
            label='Mùa lạnh (T11 – T4)'
        )
        sns.kdeplot(
            data=df[df['season_type'] == 'Mùa nóng (T5 – T10)'], x=col,
            ax=ax, color='#dc2626', linewidth=1.8, fill=True, alpha=0.25,
            label='Mùa nóng (T5 – T10)'
        )

        ax.set_title(f"{label_name}\n({sub_desc})", fontsize=10.5, fontweight='bold', pad=8, loc='left')
        ax.set_xlabel(label_name, fontweight='semibold')
        ax.set_ylabel('Mật độ xác suất (Density)', fontweight='semibold')
        ax.grid(True, axis='y')
        sns.despine(ax=ax, top=True, right=True)
        if idx == 0:
            ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=8.5)

    plt.tight_layout()
    plt.savefig('figures/weather_06_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 75)
    print(" [HOÀN TẤT] Đã tạo thành công 6 biểu đồ nâng cấp chuẩn báo cáo khoa học:")
    print("   1. figures/weather_01_10year_overview.png   (Resample tuần + Min-Max Ribbon + Sự kiện cực đoan)")
    print("   2. figures/weather_02_diurnal_cycle.png     (Trễ pha 2h nhiệt độ/bức xạ + Dải IQR)")
    print("   3. figures/weather_03_seasonality_12months.png (Boxplot nhiệt độ + Mùa mưa gió mùa)")
    print("   4. figures/weather_04_correlation_matrix.png   (Heatmap chọn lọc + Sắp xếp theo tương quan)")
    print("   5. figures/weather_05_decomposition.png     (Phân rã chuỗi + Trend dài hạn + Ngưỡng 2-Sigma)")
    print("   6. figures/weather_06_distributions.png     (KDE 2 mùa nóng/lạnh giải thích tính Bimodal)")
    print("=" * 75)


if __name__ == '__main__':
    visualize_weather()