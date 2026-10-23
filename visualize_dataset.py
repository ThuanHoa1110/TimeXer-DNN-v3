"""
========================================================================================
SCRIPT TRỰC QUAN HÓA TOÀN BỘ DỮ LIỆU ĐỀ TÀI (KAGGLE VISUALIZATION RUNNER)
Học phần: Chuyên đề Trí tuệ nhân tạo và Khoa học dữ liệu (65TTNT - Đại học Thủy Lợi)
Nhiệm vụ 2:
  1. Trực quan hóa Dataset Quy mô lớn: real_weather_large_dataset.csv (87,672 dòng, 10 năm)
  2. Trực quan hóa Dataset Bổ trợ: private_dataset.csv (15,000 dòng)
  3. Xuất toàn bộ biểu đồ chuẩn học thuật (300 DPI) vào thư mục figures/
========================================================================================
"""
import os
import sys

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi charmap
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Import module trực quan hóa khí tượng 10 năm
try:
    from visualize_weather_large import visualize_weather
except ImportError:
    visualize_weather = None


def run_private_eda():
    csv_path = "private_dataset.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "private", "private_dataset.csv")

    if not os.path.exists(csv_path):
        print(f"[!] Không tìm thấy tệp {csv_path}. Bỏ qua EDA dữ liệu webcam.")
        return

    print(f"\n[*] Đang nạp dữ liệu Private Face Biometrics từ: {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    os.makedirs("figures", exist_ok=True)
    print(f"[*] Tổng số bản ghi: {len(df):,} dòng, {df.shape[1]} cột.")

    # 1. Toàn cảnh chuỗi thời gian
    print("  [1/5] Vẽ diễn biến chuỗi thời gian Face Biometrics...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df['timestamp'], df['head_yaw'], color='#d9534f', linewidth=0.7)
    axes[0].set_ylabel('Góc quay đầu (°)\n[head_yaw]', fontsize=10, fontweight='bold')
    axes[0].set_title('DIỄN BIẾN CHUỖI THỜI GIAN HEAD POSE & FACIAL BIOMETRICS', fontsize=12, fontweight='bold')

    axes[1].plot(df['timestamp'], df['ambient_light'], color='#f0ad4e', linewidth=0.7)
    axes[1].set_ylabel('Cường độ sáng\n[ambient_light]', fontsize=10, fontweight='bold')

    axes[2].plot(df['timestamp'], df['eye_aspect_ratio'], color='#0275d8', linewidth=0.7)
    axes[2].set_ylabel('Độ mở mắt (EAR)', fontsize=10, fontweight='bold')

    axes[3].plot(df['timestamp'], df['mouth_open_ratio'], color='#5cb85c', linewidth=0.7)
    axes[3].set_ylabel('Độ mở miệng (MOR)', fontsize=10, fontweight='bold')
    axes[3].set_xlabel('Thời gian', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/private_01_overview.png', dpi=300)
    plt.close()

    # 2. Ma trận tương quan
    print("  [2/5] Tính ma trận tương quan Pearson...")
    numeric_cols = ['head_yaw', 'ambient_light', 'eye_aspect_ratio', 'mouth_open_ratio']
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(7, 5.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title('MA TRẬN TƯƠNG QUAN TARGET VÀ CÁC BIẾN NGOẠI SINH', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/private_02_correlation.png', dpi=300)
    plt.close()

    # 3. Phân phối xác suất (Histogram + KDE)
    print("  [3/5] Vẽ phân phối xác suất các đặc trưng...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    axes = axes.flatten()
    colors = ['#d9534f', '#f0ad4e', '#0275d8', '#5cb85c']
    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color=colors[idx], bins=35)
        axes[idx].set_title(f'Phân phối: {col}', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/private_03_distributions.png', dpi=300)
    plt.close()

    # 4. Xu hướng mệt mỏi theo thời gian
    print("  [4/5] Vẽ xu hướng suy giảm EAR...")
    plt.figure(figsize=(11, 4.5))
    rolling_ear = df['eye_aspect_ratio'].rolling(window=150, center=True).mean()
    plt.plot(df.index / 5.0, df['eye_aspect_ratio'], color='#b0bec5', alpha=0.5, label='EAR gốc (chớp mắt)')
    plt.plot(df.index / 5.0, rolling_ear, color='#d9534f', linewidth=2, label='EAR trung bình trượt')
    plt.axhline(y=0.20, color='red', linestyle='--', label='Ngưỡng ngủ gật')
    plt.xlabel('Thời gian mô phỏng (Giây)', fontsize=10, fontweight='bold')
    plt.ylabel('Eye Aspect Ratio (EAR)', fontsize=10, fontweight='bold')
    plt.title('PHÂN TÍCH DIỄN BIẾN ĐỘ MỞ MẮT & DẤU HIỆU BUỒN NGỦ', fontsize=11, fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/private_04_fatigue_trend.png', dpi=300)
    plt.close()

    # 5. Quan hệ giữa Head Yaw và Ambient Light
    print("  [5/5] Phân tích quan hệ chuyển động đầu và ánh sáng...")
    plt.figure(figsize=(9, 4.5))
    sns.scatterplot(data=df.iloc[::5], x='ambient_light', y='head_yaw', hue='eye_aspect_ratio', palette='viridis', alpha=0.6, s=15)
    plt.title('MỐI QUAN HỆ GIỮA GÓC QUAY ĐẦU (HEAD YAW) VÀ ÁNH SÁNG PHÒNG', fontsize=11, fontweight='bold')
    plt.xlabel('Cường độ ánh sáng (ambient_light)', fontsize=10, fontweight='bold')
    plt.ylabel('Góc quay đầu (°)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/private_05_yaw_vs_light.png', dpi=300)
    plt.close()
    print("  [✓] Hoàn tất 5 biểu đồ dữ liệu Private Face Biometrics!")


def main():
    print("=" * 70)
    print(" 📊 TIẾN HÀNH TRỰC QUAN HÓA TOÀN DIỆN DỮ LIỆU ĐỀ TÀI (KAGGLE EDA)")
    print("=" * 70)

    # 1. Chạy EDA dữ liệu khí tượng quy mô lớn (87,672 dòng, 10 năm)
    if visualize_weather is not None:
        print("\n>>> PHẦN 1: DỮ LIỆU KHÍ TƯỢNG QUY MÔ LỚN (87,672 BẢN GHI - 10 NĂM)")
        visualize_weather()
    else:
        print("[!] Không tìm thấy module visualize_weather_large.")

    # 2. Chạy EDA dữ liệu bổ trợ (15,000 dòng)
    print("\n>>> PHẦN 2: DỮ LIỆU BỔ TRỢ FACE BIOMETRICS (15,000 BẢN GHI)")
    run_private_eda()

    print("\n" + "=" * 70)
    print(" [THÀNH CÔNG] Toàn bộ biểu đồ EDA phân tích đã được xuất ra figures/:")
    print("   - figures/weather_01_10year_overview.png")
    print("   - figures/weather_02_diurnal_cycle.png")
    print("   - figures/weather_03_seasonality_12months.png")
    print("   - figures/weather_04_correlation_matrix.png")
    print("   - figures/weather_05_decomposition.png")
    print("   - figures/weather_06_distributions.png")
    print("   - figures/private_01_*.png đến private_05_*.png")
    print("   - Notebook nộp Kaggle: kaggle_eda_visualization.ipynb")
    print("=" * 70)


if __name__ == '__main__':
    main()
