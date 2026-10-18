"""
Script kiểm tra tính đúng đắn và tương thích chiều dữ liệu (tensor shape) 
của mô hình TimeXer với Backbone Encoder DNN-v3.
"""
import sys

def test_model():
    try:
        import torch
    except ImportError:
        print("[THÔNG BÁO] PyTorch chưa được cài đặt trong môi trường Python hiện tại.")
        print("Vui lòng cài đặt PyTorch (pip install torch) để chạy script kiểm tra này.")
        return

    from models.TimeXer_DnnV3 import Model as TimeXer_DnnV3_Model
    from models.TimeXer import Model as TimeXer_Model

    class DummyConfigs:
        def __init__(self, features='MS'):
            self.task_name = 'long_term_forecast'
            self.features = features
            self.seq_len = 96
            self.pred_len = 24
            self.use_norm = 1
            self.patch_len = 16
            self.enc_in = 7
            self.d_model = 128
            self.d_ff = 256
            self.dropout = 0.1
            self.activation = 'gelu'
            self.e_layers = 3
            self.embed = 'timeF'
            self.freq = 'h'
            self.encoder_type = 'dnn_v3'
            self.factor = 1
            self.n_heads = 4

    print("=================================================================")
    print(" BẮT ĐẦU KIỂM TRA MÔ HÌNH TIMEXER VỚI BACKBONE ENCODER DNN-V3")
    print("=================================================================")

    # Test 1: Multivariate predict Univariate (MS)
    print("\n--- Test 1: Nhiệm vụ MS (Multivariate predict Univariate) ---")
    cfg_ms = DummyConfigs(features='MS')
    model_ms = TimeXer_DnnV3_Model(cfg_ms)
    model_ms.eval()

    batch_size = 4
    x_enc = torch.randn(batch_size, cfg_ms.seq_len, cfg_ms.enc_in)
    x_mark_enc = torch.randn(batch_size, cfg_ms.seq_len, 4)
    x_dec = torch.randn(batch_size, cfg_ms.pred_len, cfg_ms.enc_in)
    x_mark_dec = torch.randn(batch_size, cfg_ms.pred_len, 4)

    with torch.no_grad():
        out_ms = model_ms(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print(f"Input x_enc shape: {list(x_enc.shape)}")
    print(f"Output out_ms shape: {list(out_ms.shape)}")
    expected_ms_shape = [batch_size, cfg_ms.pred_len, 1]
    assert list(out_ms.shape) == expected_ms_shape, f"Lỗi shape: mong đợi {expected_ms_shape}, nhận được {list(out_ms.shape)}"
    print("-> Test 1 PASSED: Chiều đầu ra hoàn toàn chính xác!")

    # Test 2: Multivariate predict Multivariate (M)
    print("\n--- Test 2: Nhiệm vụ M (Multivariate predict Multivariate) ---")
    cfg_m = DummyConfigs(features='M')
    model_m = TimeXer_DnnV3_Model(cfg_m)
    model_m.eval()

    with torch.no_grad():
        out_m = model_m(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print(f"Input x_enc shape: {list(x_enc.shape)}")
    print(f"Output out_m shape: {list(out_m.shape)}")
    expected_m_shape = [batch_size, cfg_m.pred_len, cfg_m.enc_in]
    assert list(out_m.shape) == expected_m_shape, f"Lỗi shape: mong đợi {expected_m_shape}, nhận được {list(out_m.shape)}"
    print("-> Test 2 PASSED: Chiều đầu ra hoàn toàn chính xác!")

    # Test 3: Kiểm tra models/TimeXer.py với cờ --encoder_type dnn_v3
    print("\n--- Test 3: Kiểm tra models/TimeXer.py với encoder_type='dnn_v3' ---")
    model_timexer = TimeXer_Model(cfg_ms)
    model_timexer.eval()
    with torch.no_grad():
        out_timexer = model_timexer(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert list(out_timexer.shape) == expected_ms_shape
    print("-> Test 3 PASSED: models/TimeXer.py hoạt động chuẩn xác với DNN-v3!")

    # Thống kê tham số
    total_params = sum(p.numel() for p in model_ms.parameters() if p.requires_grad)
    print(f"\nTổng số tham số học tập (trainable parameters): {total_params:,}")
    print("=================================================================")
    print(" TẤT CẢ CÁC BÀI TEST ĐỀU THÀNH CÔNG! SẴN SÀNG NỘP BÀI GITHUB.")
    print("=================================================================")

if __name__ == '__main__':
    test_model()
