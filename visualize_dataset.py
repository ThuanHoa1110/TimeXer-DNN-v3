"""
Script trực quan hóa bộ dữ liệu Private Time Series (private_dataset.csv)
Đề tài: Face Biometrics / Head Pose Forecasting (15,000 mẫu, 5 Hz)
Xuất các biểu đồ phân tích sang thư mục figures/
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

CSV_PATH = "private_dataset.csv"


def run_eda():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Không tìm thấy tệp {CSV_PATH}. Vui lòng chạy generate_private_data.py trước!")
        return

    print(f"[*] Đang nạp dữ liệu từ: {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    os.makedirs("figures", exist_ok=True)
    print(f"[*] Tổng số bản ghi: {len(df):,} dòng, {df.shape[1]} cột.")

    # 1. Toàn cảnh chuỗi thời gian
    print("[1/5] Đang vẽ biểu đồ diễn biến toàn cảnh chuỗi thời gian...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df['timestamp'], df['head_yaw'], color='#d9534f', linewidth=0.7)
    axes[0].set_ylabel('Góc quay đầu (°)\n[head_yaw Target]', fontsize=10, fontweight='bold')
    axes[0].set_title('DIỄN BIẾN CHUỖI THỜI GIAN DỮ LIỆU PRIVATE (HEAD POSE & FACIAL BIOMETRICS)', fontsize=13, fontweight='bold')

    axes[1].plot(df['timestamp'], df['ambient_light'], color='#f0ad4e', linewidth=0.7)
    axes[1].set_ylabel('Cường độ sáng\n[ambient_light]', fontsize=10, fontweight='bold')

    axes[2].plot(df['timestamp'], df['eye_aspect_ratio'], color='#0275d8', linewidth=0.7)
    axes[2].set_ylabel('Độ mở mắt (EAR)\n[eye_aspect_ratio]', fontsize=10, fontweight='bold')

    axes[3].plot(df['timestamp'], df['mouth_open_ratio'], color='#5cb85c', linewidth=0.7)
    axes[3].set_ylabel('Độ mở miệng (MOR)\n[mouth_open_ratio]', fontsize=10, fontweight='bold')
    axes[3].set_xlabel('Thời gian', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/private_01_overview.png', dpi=300)
    plt.close()

    # 2. Ma trận tương quan
    print("[2/5] Đang tính ma trận tương quan Pearson...")
    numeric_cols = ['head_yaw', 'ambient_light', 'eye_aspect_ratio', 'mouth_open_ratio']
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    plt.title('MA TRẬN TƯƠNG QUAN GIỮA TARGET VÀ CÁC BIẾN NGOẠI SINH', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('figures/private_02_correlation.png', dpi=300)
    plt.close()

    # 3. Phân phối xác suất (Histogram + KDE)
    print("[3/5] Đang vẽ biểu đồ phân phối xác suất các đặc trưng...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    colors = ['#d9534f', '#f0ad4e', '#0275d8', '#5cb85c']

    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color=colors[idx], bins=35)
        axes[idx].set_title(f'Phân phối: {col}', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/private_03_distributions.png', dpi=300)
    plt.close()

    # 4. Phân tích trạng thái mệt mỏi theo thời gian (EAR rolling mean)
    print("[4/5] Đang vẽ xu hướng giảm dần của EAR (Mô phỏng mệt mỏi)...")
    plt.figure(figsize=(12, 5))
    rolling_ear = df['eye_aspect_ratio'].rolling(window=150, center=True).mean()
    plt.plot(df.index / 5.0, df['eye_aspect_ratio'], color='#b0bec5', alpha=0.5, label='EAR gốc (chớp mắt)')
    plt.plot(df.index / 5.0, rolling_ear, color='#d9534f', linewidth=2, label='EAR trung bình trượt (Xu hướng mệt mỏi)')
    plt.axhline(y=0.20, color='red', linestyle='--', label='Ngưỡng ngủ gật (Drowsiness threshold)')
    plt.xlabel('Thời gian mô phỏng (Giây)', fontsize=11, fontweight='bold')
    plt.ylabel('Eye Aspect Ratio (EAR)', fontsize=11, fontweight='bold')
    plt.title('PHÂN TÍCH DIỄN BIẾN ĐỘ MỞ MẮT & DẤU HIỆU BUỒN NGỦ', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/private_04_fatigue_trend.png', dpi=300)
    plt.close()

    # 5. Phân tích mối quan hệ giữa Head Yaw và Ambient Light
    print("[5/5] Đang phân tích quan hệ giữa chuyển động đầu và ánh sáng...")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df.iloc[::5],  # subsample để vẽ nhanh
        x='ambient_light',
        y='head_yaw',
        hue='eye_aspect_ratio',
        palette='viridis',
        alpha=0.6,
        s=15
    )
    plt.title('MỐI QUAN HỆ GIỮA GÓC QUAY ĐẦU (HEAD YAW) VÀ ÁNH SÁNG PHÒNG', fontsize=12, fontweight='bold')
    plt.xlabel('Cường độ ánh sáng (ambient_light)', fontsize=11, fontweight='bold')
    plt.ylabel('Góc quay đầu (°)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/private_05_yaw_vs_light.png', dpi=300)
    plt.close()

    print("\n" + "=" * 65)
    print(" [HOÀN TẤT] Đã tạo thành công 5 biểu đồ cho private_dataset.csv:")
    print("   1. figures/private_01_overview.png")
    print("   2. figures/private_02_correlation.png")
    print("   3. figures/private_03_distributions.png")
    print("   4. figures/private_04_fatigue_trend.png")
    print("   5. figures/private_05_yaw_vs_light.png")
    print("=" * 65)


if __name__ == '__main__':
    run_eda()
