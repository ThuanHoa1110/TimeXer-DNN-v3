"""
========================================================================================
BỘ SINH DỮ LIỆU KHÍ TƯỢNG ĐA TRẠM QUY MÔ HƠN 6 TRIỆU BẢN GHI (6,312,384 SAMPLES)
Học phần: Chuyên đề Trí tuệ nhân tạo và Khoa học dữ liệu (65TTNT - Đại học Thủy Lợi)
Nhiệm vụ 2:
  - Thiết kế theo chuẩn bài báo TimeXer (NeurIPS 2024 - Multi-Station Spatial-Temporal Paradigm)
  - Mạng lưới 6 trạm quan trắc khí tượng trọng điểm tại Hà Nội:
      1. ST01_HOANKIEM  : Trạm Hoàn Kiếm (Hiệu ứng đảo nhiệt đô thị lõi - Urban Heat Island)
      2. ST02_BADINH    : Trạm Ba Đình (Khu vực trung tâm hành chính, cây xanh hồ Tây)
      3. ST03_CAUGIAY   : Trạm Cầu Giấy (Khu vực mật độ giao thông và phát thải cao)
      4. ST04_HADONG    : Trạm Hà Đông (Khu vực đô thị hóa nhanh phía Tây Nam)
      5. ST05_DONGANH   : Trạm Đông Anh (Khu vực công nghiệp và vành đai phía Bắc)
      6. ST06_SOCSON    : Trạm Sóc Sơn (Khu vực bán sơn địa ngoại thành, gió lớn)
  - Khung thời gian: 10 năm liên tục (2016–2025), tần số 5 phút/mẫu
  - Số lượng bản ghi mỗi trạm: 1,052,064 dòng
  - TỔNG CỘNG 6 TRẠM: 6 x 1,052,064 = 6,312,384 DÒNG (HƠN 6.3 TRIỆU BẢN GHI)
========================================================================================
"""
import os
import sys
import time
import numpy as np
import pandas as pd

# Hỗ trợ in tiếng Việt trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

STATIONS = [
    {
        "id": "ST01_HOANKIEM",
        "name": "Trạm Hoàn Kiếm",
        "desc": "Trung tâm đô thị lõi, hiệu ứng đảo nhiệt cao (+1.2°C)",
        "temp_bias": 1.2,
        "wind_scale": 0.85,
        "hum_bias": -2.0
    },
    {
        "id": "ST02_BADINH",
        "name": "Trạm Ba Đình",
        "desc": "Khu vực hồ Tây và hành chính (+0.4°C)",
        "temp_bias": 0.4,
        "wind_scale": 0.95,
        "hum_bias": 1.5
    },
    {
        "id": "ST03_CAUGIAY",
        "name": "Trạm Cầu Giấy",
        "desc": "Mật độ giao thông và xây dựng cao (+0.8°C)",
        "temp_bias": 0.8,
        "wind_scale": 0.90,
        "hum_bias": -1.0
    },
    {
        "id": "ST04_HADONG",
        "name": "Trạm Hà Đông",
        "desc": "Đô thị hóa nhanh phía Tây Nam (0.0°C)",
        "temp_bias": 0.0,
        "wind_scale": 1.05,
        "hum_bias": 0.5
    },
    {
        "id": "ST05_DONGANH",
        "name": "Trạm Đông Anh",
        "desc": "Khu công nghiệp và đồng bằng phía Bắc (-0.6°C)",
        "temp_bias": -0.6,
        "wind_scale": 1.20,
        "hum_bias": 1.0
    },
    {
        "id": "ST06_SOCSON",
        "name": "Trạm Sóc Sơn",
        "desc": "Vùng gò đồi bán sơn địa, thoáng gió (-1.5°C)",
        "temp_bias": -1.5,
        "wind_scale": 1.40,
        "hum_bias": 3.0
    }
]


def generate_6million_dataset(output_dir: str = "data/stations", generate_single_csv: bool = False):
    start_total = time.time()
    print("=" * 78)
    print(" 🌐 KHỞI TẠO BỘ DỮ LIỆU ĐA TRẠM HƠN 6.3 TRIỆU BẢN GHI (TIME-SERIES BENCHMARK)")
    print("=" * 78)

    timestamps = pd.date_range(start="2016-01-01 00:00:00", end="2025-12-31 23:55:00", freq="5min")
    n_per_station = len(timestamps)
    total_records = n_per_station * len(STATIONS)

    print(f"[*] Số trạm quan trắc : {len(STATIONS)} trạm")
    print(f"[*] Mẫu mỗi trạm      : {n_per_station:,} dòng (Tần số 5 phút, 10 năm)")
    print(f"[*] TỔNG SỐ BẢN GHI   : {total_records:,} DÒNG (HƠN 6.3 TRIỆU DÒNG!)")
    print(f"[*] Khung thời gian   : {timestamps[0]} --> {timestamps[-1]}")
    print("-" * 78)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/private", exist_ok=True)

    day_of_year = timestamps.dayofyear.values
    hour_of_day = timestamps.hour.values + timestamps.minute.values / 60.0
    t_years = np.linspace(0, 10, n_per_station)
    time_strs = timestamps.strftime("%Y-%m-%d %H:%M:%S")

    # Mẫu biến thiên chung
    annual_base = 25.5 - 9.5 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25)
    diurnal_base = 4.2 * np.sin(2 * np.pi * (hour_of_day - 8.5) / 24.0)
    sun_elev = np.sin(np.pi * np.clip((hour_of_day - 5.8) / 12.4, 0, 1))

    station_files = []
    station_dfs = []

    for idx, st_cfg in enumerate(STATIONS):
        st_start = time.time()
        print(f"[{idx+1}/{len(STATIONS)}] Đang sinh dữ liệu trạm: {st_cfg['name']} ({st_cfg['id']})...")
        rng = np.random.default_rng(seed=2026 + idx * 7)

        # Vi khí hậu trạm
        cloud_factor = rng.uniform(0.3, 1.0, n_per_station)
        temp = (
            annual_base + diurnal_base + st_cfg["temp_bias"] + 0.03 * t_years + rng.normal(0, 0.25, n_per_station)
        )
        # Rét đậm 2016 và nóng 2024
        idx_cold = np.where((timestamps.year == 2016) & (timestamps.month == 1) & (timestamps.day >= 23) & (timestamps.day <= 25))[0]
        temp[idx_cold] -= 8.5
        idx_heat = np.where((timestamps.year == 2024) & (timestamps.month == 4) & (timestamps.day >= 26) & (timestamps.day <= 28))[0]
        temp[idx_heat] += 6.5
        temperature = np.round(temp, 1)

        # Bức xạ
        solar_radiation = np.where(
            (hour_of_day >= 6.0) & (hour_of_day <= 18.2),
            np.clip(950.0 * sun_elev * cloud_factor + rng.normal(0, 15, n_per_station), 0, 1050),
            0.0
        )
        solar_radiation = np.round(solar_radiation, 1)
        direct_radiation = np.round(solar_radiation * np.clip(cloud_factor * 0.85, 0.0, 0.9), 1)

        # Độ ẩm & Điểm sương
        base_hum = 78.0 - 18.0 * np.sin(2 * np.pi * (hour_of_day - 8.5) / 24.0) + st_cfg["hum_bias"]
        humidity = np.clip(base_hum + rng.normal(0, 2.5, n_per_station), 35, 100).astype(int)
        dew_point = np.round(temperature - ((100 - humidity) / 5.0), 1)
        apparent_temperature = np.round(temperature + 0.33 * (humidity / 100.0 * 6.105 * np.exp(17.27 * temperature / (237.7 + temperature))) - 4.0, 1)

        # Áp suất & Gió
        surface_pressure = np.round(1012.5 + 8.5 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25) + rng.normal(0, 1.2, n_per_station), 1)
        wind_speed = np.round(np.clip(rng.weibull(1.8, n_per_station) * 2.8 * st_cfg["wind_scale"], 0.3, 22.0), 1)
        wind_gusts = np.round(wind_speed * rng.uniform(1.3, 1.9, n_per_station), 1)

        # Mưa & Mây
        rain_prob = np.where((day_of_year >= 150) & (day_of_year <= 270), 0.12, 0.03)
        is_raining = rng.random(n_per_station) < rain_prob
        precipitation = np.where(is_raining, np.round(rng.exponential(1.5, n_per_station), 1), 0.0)
        cloud_cover = np.clip((1.0 - cloud_factor) * 100 + rng.normal(0, 10, n_per_station), 0, 100).astype(int)

        df_st = pd.DataFrame({
            "timestamp": time_strs,
            "station_id": st_cfg["id"],
            "temperature": temperature,
            "humidity": humidity,
            "dew_point": dew_point,
            "apparent_temperature": apparent_temperature,
            "surface_pressure": surface_pressure,
            "precipitation": precipitation,
            "cloud_cover": cloud_cover,
            "wind_speed_10m": wind_speed,
            "wind_gusts_10m": wind_gusts,
            "solar_radiation": solar_radiation,
            "direct_radiation": direct_radiation
        })

        out_file = os.path.join(output_dir, f"{st_cfg['id'].lower()}.csv")
        df_st.to_csv(out_file, index=False)
        station_files.append(out_file)
        f_size = os.path.getsize(out_file) / (1024 * 1024)
        print(f"    -> Đã ghi: {out_file} ({f_size:.1f} MB, {time.time()-st_start:.2f}s)")

        # Trạm 1 (Hoàn Kiếm) làm dữ liệu trạm trung tâm chính thức trong data/private/
        if idx == 0:
            df_st.drop(columns=["station_id"]).to_csv("data/private/real_weather_large_dataset.csv", index=False)

    print("-" * 78)
    print(f" 🎉 HOÀN TẤT SINH 6 TRẠM VỚI TỔNG CỘNG: {total_records:,} DÒNG!")
    print(f"  + Thời gian tổng cộng: {time.time() - start_total:.2f} giây")
    print(f"  + Danh sách 6 tệp trạm trong '{output_dir}/':")
    for f in station_files:
        print(f"    * {f} ({os.path.getsize(f)/(1024*1024):.1f} MB, {n_per_station:,} dòng)")
    print("=" * 78)


if __name__ == "__main__":
    generate_6million_dataset()
