"""
========================================================================================
AI CI/CD UNIT TEST SUITE - TIMEXER-DINOv3 WEATHER AI
Học phần: Chuyên đề Trí tuệ Nhân tạo và Khoa học Dữ liệu (Lớp 65TTNT - ĐH Thủy Lợi)
Sinh viên: Nguyễn Thuận Hóa

Các ca kiểm thử tự động (Automated Verification Cases):
  1. Test Data Integrity & Schema Validation (Dữ liệu trạm khí tượng mẫu)
  2. Test TimeXer-DINOv3 Model Architecture & Tensor Flow (Shape MS & Attention Heads)
  3. Test DINO-v3 Composite Loss Function & Gradient Backward Flow
  4. Test Model Serving Inference Engine & Checkpoint Loading
  5. Test Web API REST Endpoint Serialization
========================================================================================
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class TestAICICDPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = BASE_DIR
        cls.sample_station_path = os.path.join(BASE_DIR, "data", "stations_sample", "st01_hoankiem.csv")
        cls.checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "timexer_dinov3_best.pth")
        cls.scaler_path = os.path.join(BASE_DIR, "checkpoints", "scaler_params.json")

    # -------------------------------------------------------------------------
    # TEST 1: Kiểm thử tính toàn vẹn của Dữ liệu đầu vào (Data Integrity Test)
    # -------------------------------------------------------------------------
    def test_01_data_integrity_and_schema(self):
        """Kiểm tra tệp dữ liệu mẫu quan trắc khí tượng có đúng cấu trúc schema và không chứa NaN"""
        self.assertTrue(os.path.exists(self.sample_station_path), f"Không tìm thấy file trạm: {self.sample_station_path}")
        df = pd.read_csv(self.sample_station_path)
        
        # Kiểm tra tối thiểu 96 bước thời gian (Lookback window)
        self.assertGreaterEqual(len(df), 96, "Dữ liệu quan trắc trạm phải có ít nhất 96 mốc thời gian.")
        
        # Kiểm tra các cột đặc trưng bắt buộc
        expected_cols = ['timestamp', 'temperature', 'humidity', 'solar_radiation', 'wind_speed_10m', 'surface_pressure']
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Thiếu cột dữ liệu bắt buộc: {col}")

        # Kiểm tra không có NaN ở 96 dòng cuối
        recent_df = df.iloc[-96:]
        self.assertEqual(recent_df[expected_cols[1:]].isna().sum().sum(), 0, "Dữ liệu chuỗi thời gian đầu vào không được chứa NaN.")

    # -------------------------------------------------------------------------
    # TEST 2: Kiểm thử Kiến trúc Mô hình TimeXer-DINOv3 (Model Architecture Test)
    # -------------------------------------------------------------------------
    def test_02_dinov3_architecture_forward(self):
        """Kiểm tra forward pass qua DINO-v3 Backbone với LayerScale và Cross-Attention"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch chưa được cài đặt trong môi trường.")

        from models.TimeXer_DINOv3 import Model as TimeXer_DINOv3_Model

        class DummyConfig:
            task_name = 'long_term_forecast'
            features = 'MS'
            seq_len = 96
            pred_len = 24
            use_norm = 1
            patch_len = 16
            enc_in = 5
            d_model = 64
            d_ff = 128
            dropout = 0.05
            activation = 'gelu'
            e_layers = 2
            n_heads = 4
            embed = 'timeF'
            freq = '5min'
            encoder_type = 'dino_v3'
            factor = 1
            dino_temp_s = 0.1
            dino_temp_t = 0.04
            dino_lambda_distill = 0.1
            dino_lambda_gram = 0.05

        cfg = DummyConfig()
        model = TimeXer_DINOv3_Model(cfg)
        model.eval()

        batch_size = 2
        x_enc = torch.randn(batch_size, cfg.seq_len, cfg.enc_in)
        x_mark_enc = torch.randn(batch_size, cfg.seq_len, 4)
        x_dec = torch.randn(batch_size, cfg.pred_len, cfg.enc_in)
        x_mark_dec = torch.randn(batch_size, cfg.pred_len, 4)

        with torch.no_grad():
            pred, dino_proj, patch_tokens = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_dino=True)

        self.assertEqual(list(pred.shape), [batch_size, cfg.pred_len, 1], "Đầu ra dự báo MS phải có shape [Batch, Pred_Len, 1]")
        self.assertEqual(pred.shape[0], batch_size)
        self.assertFalse(torch.isnan(pred).any(), "Kết quả dự báo chứa giá trị NaN!")

    # -------------------------------------------------------------------------
    # TEST 3: Kiểm thử Hàm mất mát DINO-v3 Composite Loss & Backward Gradient
    # -------------------------------------------------------------------------
    def test_03_dinov3_composite_loss_and_gradients(self):
        """Kiểm tra 3 thành phần hàm mất mát DINOv3 (MSE + Distill + Gram) và gradient backward lan truyền thành công"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch chưa được cài đặt trong môi trường.")

        from models.TimeXer_DINOv3 import Model as TimeXer_DINOv3_Model, DINOv3Loss

        class DummyConfig:
            task_name = 'long_term_forecast'
            features = 'MS'
            seq_len = 96
            pred_len = 24
            use_norm = 1
            patch_len = 16
            enc_in = 5
            d_model = 64
            d_ff = 128
            dropout = 0.05
            activation = 'gelu'
            e_layers = 1
            n_heads = 2
            embed = 'timeF'
            freq = '5min'
            encoder_type = 'dino_v3'
            factor = 1
            dino_temp_s = 0.1
            dino_temp_t = 0.04
            dino_lambda_distill = 0.1
            dino_lambda_gram = 0.05

        cfg = DummyConfig()
        model = TimeXer_DINOv3_Model(cfg)
        model.train()

        loss_fn = DINOv3Loss(
            task_criterion=nn.MSELoss(),
            student_temp=0.1,
            teacher_temp=0.04,
            lambda_distill=0.1,
            lambda_gram=0.05,
            out_dim=256
        )

        batch_size = 2
        x_enc = torch.randn(batch_size, cfg.seq_len, cfg.enc_in)
        x_mark_enc = torch.randn(batch_size, cfg.seq_len, 4)
        x_dec = torch.randn(batch_size, cfg.pred_len, cfg.enc_in)
        x_mark_dec = torch.randn(batch_size, cfg.pred_len, 4)
        target_y = torch.randn(batch_size, cfg.pred_len, 1)

        pred, dino_proj, patch_tokens = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_dino=True)
        teacher_proj = dino_proj.detach() + torch.randn_like(dino_proj) * 0.01

        loss_total, loss_dict = loss_fn(pred, target_y, dino_proj, teacher_proj, patch_tokens)
        self.assertGreater(float(loss_total.item()), 0.0)
        self.assertIn('loss_task', loss_dict)
        self.assertIn('loss_distill', loss_dict)
        self.assertIn('loss_gram', loss_dict)

        # Kiểm tra Gradient Backward
        loss_total.backward()
        has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
        self.assertTrue(has_grad, "Gradient backward lan truyền thất bại hoặc bị triệt tiêu hoàn toàn!")

    # -------------------------------------------------------------------------
    # TEST 4: Kiểm thử Nạp Checkpoint & Inference Engine
    # -------------------------------------------------------------------------
    def test_04_inference_engine_prediction(self):
        """Kiểm tra TimeXerInferenceEngine nạp checkpoint thật và thực thi suy luận ra 24 bước dự báo"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch chưa được cài đặt trong môi trường.")

        from inference_engine import TimeXerInferenceEngine

        engine = TimeXerInferenceEngine()
        self.assertIsNotNone(engine.model, "Mô hình trong Inference Engine chưa được khởi tạo.")

        df_input = pd.read_csv(self.sample_station_path)
        result = engine.predict(df_input=df_input, horizon=24, use_exogenous=True)

        self.assertIn('predictions', result)
        self.assertIn('lookback_history', result)
        self.assertIn('metrics', result)
        self.assertEqual(len(result['predictions']), 24, "Độ dài kết quả dự báo phải đúng bằng 24 bước.")
        self.assertIn('predicted_temperature', result['predictions'].columns)

    # -------------------------------------------------------------------------
    # TEST 5: Kiểm thử Model Serving Payload Serialization
    # -------------------------------------------------------------------------
    def test_05_api_payload_formatting(self):
        """Kiểm tra tính tương thích dữ liệu JSON serialize phục vụ REST API Serving"""
        from inference_engine import TimeXerInferenceEngine

        engine = TimeXerInferenceEngine()
        df_input = pd.read_csv(self.sample_station_path)
        result = engine.predict(df_input=df_input, horizon=24, use_exogenous=True)

        response_payload = {
            "status": "success",
            "model": "TimeXer_DINOv3",
            "lookback_history": result['lookback_history'][['timestamp', 'temperature']].to_dict(orient='records'),
            "predictions": result['predictions'].to_dict(orient='records'),
            "metrics": result['metrics'],
            "exogenous_attribution": result['exogenous_attribution']
        }

        # Thử json.dumps xem có serialize được không
        serialized = json.dumps(response_payload, default=str)
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(serialized), 100)


if __name__ == '__main__':
    unittest.main()
