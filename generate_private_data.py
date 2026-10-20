"""
Script tao du lieu chuoi thoi gian Private chuan (Private Time Series Dataset)
phuc vu bai toan Du bao bien ngoai sinh (Time Series Forecasting with Exogenous Variables)
theo mo hinh TimeXer - Lop 65TTNT.

De tai: Face Biometrics / Head Pose Forecasting
- Bien noi sinh (Endogenous Target): head_yaw (goc quay dau theo do, dai luong can du bao)
- Bien ngoai sinh (Exogenous Covariates):
    + ambient_light: cuong do anh sang moi truong (0.1 - 1.0)
    + eye_aspect_ratio: do mo mat (chi so tinh tao / met moi)
    + mouth_open_ratio: do mo mieng (noi chuyen / ngap)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_private_dataset(
    output_path: str = "private_dataset.csv",
    n_steps: int = 15000,
    fps: float = 5.0,
    seed: int = 42,
):
    print(f"Dang khoi tao du lieu Private Time Series gom {n_steps} mau (tan so {fps} Hz)...")
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) / fps

    # 1. Bien ngoai sinh: ambient_light (Bien thien theo anh sang phong va thoi gian trong ngay)
    light_base = 0.65 + 0.15 * np.sin(2 * np.pi * t / 600)
    light_noise = rng.normal(0, 0.02, n_steps)
    ambient_light = np.clip(light_base + light_noise, 0.15, 1.0)

    # 2. Bien noi sinh: head_yaw (Goc quay dau: gom chuyen dong lac dau cham + cu dong nhanh + nhieu)
    slow_drift = 18.0 * np.sin(2 * np.pi * t / 90) + 10.0 * np.cos(2 * np.pi * t / 25)
    saccades = np.zeros(n_steps)
    # Mo phong cac cu dong quay dau dot ngot
    num_saccades = n_steps // 120
    for _ in range(num_saccades):
        idx = rng.integers(10, n_steps - 30)
        width = rng.integers(5, 25)
        saccades[idx : idx + width] += rng.uniform(-25.0, 25.0)

    yaw_noise = rng.normal(0, 1.5 + (1.0 - ambient_light) * 2.5, n_steps)
    head_yaw = slow_drift + saccades + yaw_noise

    # 3. Bien ngoai sinh: eye_aspect_ratio (EAR - do mo mat giam dan khi lam viec lau / met moi)
    fatigue_trend = np.linspace(0, 0.08, n_steps)
    eye_aspect_ratio = (
        0.30 - fatigue_trend + 0.02 * np.sin(2 * np.pi * t / 45) + rng.normal(0, 0.01, n_steps)
    )
    eye_aspect_ratio = np.clip(eye_aspect_ratio, 0.12, 0.38)

    # 4. Bien ngoai sinh: mouth_open_ratio (Do mo mieng tang khi noi chuyen hoac thay doi tu the)
    mouth_base = 0.04 + 0.02 * np.abs(np.diff(head_yaw, prepend=head_yaw[0])) / 10.0
    mouth_open_ratio = np.clip(mouth_base + rng.normal(0, 0.008, n_steps), 0.0, 0.35)

    # Tao moc thoi gian lien tuc
    timestamps = pd.date_range("2026-10-01 08:00:00", periods=n_steps, freq=f"{int(1000/fps)}ms")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "head_yaw": np.round(head_yaw.astype(np.float32), 4),
            "ambient_light": np.round(ambient_light.astype(np.float32), 4),
            "eye_aspect_ratio": np.round(eye_aspect_ratio.astype(np.float32), 4),
            "mouth_open_ratio": np.round(mouth_open_ratio.astype(np.float32), 4),
        }
    )

    # Luu file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"-> Da tao thanh cong file: {os.path.abspath(output_path)}")
    print(f"-> So dong: {len(df):,}, So cot: {df.shape[1]}")
    print(f"-> Danh sach cot: {list(df.columns)}")
    print("\n5 dong dau tien:")
    print(df.head())
    return df


if __name__ == "__main__":
    # Tao file private_dataset.csv ngay tai thu muc goc
    generate_private_dataset("private_dataset.csv")
    # Dong thoi tao ban sao trong data/private/ de dong bo thu muc du an
    generate_private_dataset("data/private/private_dataset.csv")
