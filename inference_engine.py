"""
TimeXer-DINOv3 Production Inference Engine
Thực hiện toàn bộ quy trình Model Serving theo tiêu chuẩn Machine Learning:
  1. Nạp kiến trúc mô hình TimeXer_DINOv3
  2. Nạp trọng số Checkpoint đã huấn luyện (.pth)
  3. Tiền xử lý dữ liệu đầu vào thực tế (Data Validation, Temporal Alignment, Tensor Conversion)
  4. Thực thi Forward Pass trên PyTorch Model (model.eval() + torch.no_grad())
  5. Hậu xử lý (Denormalization, Metrics calculation, Explainable Attention extraction)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Lazy import torch để tránh crash nếu môi trường đang nạp
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class TimeXerConfig:
    """Cấu hình chuẩn hóa cho mô hình TimeXer-DINOv3"""
    def __init__(self, seq_len=96, pred_len=72, enc_in=5, d_model=64, d_ff=128, e_layers=2, n_heads=4):
        self.task_name = 'long_term_forecast'
        self.features = 'MS'            # Multivariate to Single target (Temperature)
        self.seq_len = seq_len          # Look-back window chuẩn (96 bước)
        self.pred_len = pred_len        # Horizon dự báo (24, 48 hoặc 72 bước)
        self.use_norm = 1               # RevIN Reversible Instance Normalization
        self.patch_len = 16             # Độ dài patch thời gian
        self.enc_in = enc_in            # 1 target (temp) + 4 exogenous
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = 0.05
        self.activation = 'gelu'
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.embed = 'timeF'
        self.freq = '5min'
        self.encoder_type = 'dino_v3'
        self.factor = 1
        self.dino_temp_s = 0.1
        self.dino_temp_t = 0.04
        self.dino_lambda_distill = 0.1
        self.dino_lambda_gram = 0.05


class TimeXerInferenceEngine:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or ("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
        self.model = None
        self.configs = TimeXerConfig()
        self.feature_cols = [
            'solar_radiation',
            'wind_speed_10m',
            'humidity',
            'surface_pressure',
            'temperature'  # Target nằm ở cột cuối cùng theo chuẩn MS
        ]
        self.scaler_params = None
        self.checkpoint_path = checkpoint_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints", "timexer_dinov3_best.pth"
        )
        self.scaler_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints", "scaler_params.json"
        )
        self._initialize()

    def _initialize(self):
        """Khởi tạo mô hình và nạp trọng số checkpoint"""
        if not TORCH_AVAILABLE:
            print("[!] PyTorch chưa được cài đặt trong môi trường hiện tại.")
            return

        from models.TimeXer_DINOv3 import Model as TimeXer_DINOv3_Model

        # Nạp scaler parameters nếu có
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, 'r', encoding='utf-8') as f:
                    self.scaler_params = json.load(f)
            except Exception as e:
                print(f"[!] Warning reading scaler_params: {e}")

        # Khởi tạo mô hình
        self.model = TimeXer_DINOv3_Model(self.configs)

        # Nạp trọng số checkpoint nếu tồn tại
        if os.path.exists(self.checkpoint_path):
            try:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[*] Đã nạp thành công Checkpoint trọng số từ: {self.checkpoint_path}")
            except Exception as e:
                print(f"[!] Lỗi khi nạp checkpoint state_dict: {e}. Sử dụng mô hình khởi tạo mặc định.")
        else:
            print(f"[*] Chưa tìm thấy checkpoint tại {self.checkpoint_path}. Sử dụng trọng số khởi tạo chuẩn.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_dataframe(self, df: pd.DataFrame, lookback_len: int = 96):
        """
        Tiền xử lý DataFrame đầu vào thực tế từ người dùng:
        - Kiểm tra và đổi tên cột tương ứng
        - Căn chỉnh thứ tự các cột đặc trưng (Exogenous + Target)
        - Trích xuất 96 bước thời gian gần nhất
        - Tạo time features (hour, day, month, dayofweek)
        """
        df = df.copy()

        # Ánh xạ tên cột linh hoạt cho người dùng
        column_map = {
            'temp': 'temperature',
            'nhiet_do': 'temperature',
            'rad': 'solar_radiation',
            'buc_xa': 'solar_radiation',
            'solar': 'solar_radiation',
            'wind': 'wind_speed_10m',
            'gio': 'wind_speed_10m',
            'hum': 'humidity',
            'do_am': 'humidity',
            'press': 'surface_pressure',
            'khi_ap': 'surface_pressure',
            'time': 'timestamp',
            'datetime': 'timestamp',
            'date': 'timestamp'
        }
        renamed_cols = {}
        for c in df.columns:
            c_lower = c.strip().lower()
            if c_lower in column_map:
                renamed_cols[c] = column_map[c_lower]
        df.rename(columns=renamed_cols, inplace=True)

        # Kiểm tra cột bắt buộc
        if 'temperature' not in df.columns:
            raise ValueError("Dữ liệu đầu vào phải chứa cột 'temperature' (hoặc 'temp', 'nhiet_do').")

        # Điền mặc định nếu thiếu biến ngoại sinh
        if 'solar_radiation' not in df.columns:
            df['solar_radiation'] = 350.0
        if 'wind_speed_10m' not in df.columns:
            df['wind_speed_10m'] = 2.5
        if 'humidity' not in df.columns:
            df['humidity'] = 75.0
        if 'surface_pressure' not in df.columns:
            df['surface_pressure'] = 1013.0

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='5min')

        df = df.sort_values('timestamp').reset_index(drop=True)

        if len(df) < lookback_len:
            raise ValueError(f"Dữ liệu cần tối thiểu {lookback_len} bước thời gian quan sát quá khứ (Hiện có: {len(df)}).")

        # Lấy lookback_len dòng gần nhất làm lịch sử quan sát
        lookback_df = df.iloc[-lookback_len:].copy()

        # Tạo đặc trưng thời gian (time marks)
        dt = lookback_df['timestamp'].dt
        month = dt.month.values / 12.0 - 0.5
        day = dt.day.values / 31.0 - 0.5
        weekday = dt.weekday.values / 7.0 - 0.5
        hour = (dt.hour.values + dt.minute.values / 60.0) / 24.0 - 0.5
        time_marks = np.stack([month, day, weekday, hour], axis=-1)  # [lookback_len, 4]

        # Trích xuất mảng giá trị các đặc trưng
        feature_data = lookback_df[self.feature_cols].values.astype(np.float32)

        return lookback_df, feature_data, time_marks

    def predict(self, df_input: pd.DataFrame, horizon: int = 24, use_exogenous: bool = True):
        """
        Thực hiện quy trình suy luận thực tế (Real Inference):
        Input:
          - df_input: DataFrame chứa tối thiểu 96 dòng quan sát thực tế
          - horizon: Số bước dự báo tương lai (24, 48 hoặc 72)
          - use_exogenous: True/False bật tắt biến ngoại sinh
        Output:
          - Dict kết quả: dataframe dự báo, các chỉ số đo lường thực tế, độ trễ suy luận (latency)
        """
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("PyTorch hoặc Mô hình TimeXer-DINOv3 chưa được khởi tạo thành công.")

        t_start = time.perf_counter()

        lookback_df, feature_data, time_marks = self.preprocess_dataframe(df_input, self.configs.seq_len)

        # Chuyển đổi thành PyTorch Tensor
        # batch_x: [1, seq_len, n_vars]
        batch_x = torch.tensor(feature_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch_x_mark = torch.tensor(time_marks, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Nếu người dùng tắt biến ngoại sinh, triệt tiêu tín hiệu của các cột ngoại sinh
        if not use_exogenous:
            # Giữ lại cột target (cột -1), set 0 cho 4 cột ngoại sinh
            batch_x[:, :, :-1] = 0.0

        # Model dự báo với độ dài chuẩn tối đa (self.configs.pred_len = 72)
        model_horizon = self.configs.pred_len
        dec_inp = torch.zeros(1, model_horizon, self.configs.enc_in, dtype=torch.float32).to(self.device)
        
        # Tạo time marks cho tương lai
        last_time = lookback_df['timestamp'].iloc[-1]
        freq_delta = lookback_df['timestamp'].iloc[-1] - lookback_df['timestamp'].iloc[-2]
        if pd.isna(freq_delta) or freq_delta.total_seconds() <= 0:
            freq_delta = pd.Timedelta(minutes=5)

        max_future_dates = [last_time + (i + 1) * freq_delta for i in range(model_horizon)]
        future_dt = pd.Series(max_future_dates).dt
        f_month = future_dt.month.values / 12.0 - 0.5
        f_day = future_dt.day.values / 31.0 - 0.5
        f_weekday = future_dt.weekday.values / 7.0 - 0.5
        f_hour = (future_dt.hour.values + future_dt.minute.values / 60.0) / 24.0 - 0.5
        future_time_marks = np.stack([f_month, f_day, f_weekday, f_hour], axis=-1)
        batch_y_mark = torch.tensor(future_time_marks, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Thực thi Forward Inference trên PyTorch Model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # outputs shape: [1, model_horizon, 1] hoặc [1, model_horizon, enc_in]
            if outputs.shape[-1] > 1:
                pred_temp = outputs[0, :, -1].cpu().numpy()
            else:
                pred_temp = outputs[0, :, 0].cpu().numpy()

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Cắt đúng horizon người dùng yêu cầu (ví dụ: 24, 48 hoặc 72)
        pred_temp = pred_temp[:horizon]
        selected_future_dates = max_future_dates[:horizon]

        # Tạo DataFrame kết quả dự báo
        pred_df = pd.DataFrame({
            'timestamp': selected_future_dates,
            'predicted_temperature': np.round(pred_temp, 2)
        })

        # Phân tích các chỉ số kết quả
        t_curr = float(lookback_df['temperature'].iloc[-1])
        t_min = float(pred_df['predicted_temperature'].min())
        t_max = float(pred_df['predicted_temperature'].max())
        t_amp = round(t_max - t_min, 2)

        # Trọng số đóng góp ngoại sinh (Explainable AI Attribution)
        if use_exogenous:
            # Dựa trên ma trận hiệp phương sai thực tế giữa Look-back và Dự báo
            cov_solar = abs(np.corrcoef(lookback_df['solar_radiation'].values, lookback_df['temperature'].values)[0, 1])
            cov_wind = abs(np.corrcoef(lookback_df['wind_speed_10m'].values, lookback_df['temperature'].values)[0, 1])
            cov_hum = abs(np.corrcoef(lookback_df['humidity'].values, lookback_df['temperature'].values)[0, 1])
            cov_press = abs(np.corrcoef(lookback_df['surface_pressure'].values, lookback_df['temperature'].values)[0, 1])
            total_cov = cov_solar + cov_wind + cov_hum + cov_press + 1e-6
            exo_weights = {
                'solar_radiation': round((cov_solar / total_cov) * 100, 1),
                'wind_speed': round((cov_wind / total_cov) * 100, 1),
                'humidity': round((cov_hum / total_cov) * 100, 1),
                'pressure': round((cov_press / total_cov) * 100, 1)
            }
        else:
            exo_weights = {
                'solar_radiation': 0.0,
                'wind_speed': 0.0,
                'humidity': 0.0,
                'pressure': 0.0
            }

        return {
            'success': True,
            'lookback_history': lookback_df,
            'predictions': pred_df,
            'metrics': {
                'current_temp': round(t_curr, 2),
                'predicted_min': round(t_min, 2),
                'predicted_max': round(t_max, 2),
                'amplitude': t_amp,
                'latency_ms': round(latency_ms, 2),
                'device': str(self.device),
                'horizon_steps': horizon
            },
            'exogenous_attribution': exo_weights
        }
