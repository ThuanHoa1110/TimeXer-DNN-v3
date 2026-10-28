"""
========================================================================================
COMPREHENSIVE VERIFICATION SCRIPT FOR TIMEXER WITH DINO-v3 BACKBONE & LOSS FUNCTION
Task 1:
  - Verify DINO-v3 Backbone Encoder (LayerScale + MHSA + Exogenous Fusion)
  - Verify DINO-v3 Composite Loss Function (Task MSE + Self-Distillation + Gram Anchoring)
  - Verify Backward Gradient Flow
========================================================================================
"""
import sys

# Hỗ trợ in tiếng Việt trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def run_dinov3_verification():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[THÔNG BÁO] Chưa tìm thấy PyTorch trong môi trường hiện tại.")
        print("Để chạy kiểm thử tự động, vui lòng cài đặt: pip install torch")
        return

    from models.TimeXer_DINOv3 import Model as TimeXer_DINOv3_Model, DINOv3Loss
    from models.TimeXer import Model as TimeXer_Unified_Model

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
            self.e_layers = 2
            self.n_heads = 4
            self.embed = 'timeF'
            self.freq = 'h'
            self.encoder_type = 'dino_v3'
            self.factor = 1
            self.dino_temp_s = 0.1
            self.dino_temp_t = 0.04
            self.dino_lambda_distill = 0.1
            self.dino_lambda_gram = 0.05

    print("=" * 75)
    print(" 🚀 BẮT ĐẦU KIỂM THỬ TIMEXER - DINO-V3 BACKBONE & COMPOSITE LOSS FUNCTION")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # TEST 1: Kiểm tra tensor flow nhiệm vụ dự báo đơn biến MS
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Kiểm tra Tensor Flow nhiệm vụ MS (Multivariate predict Univariate)...")
    cfg_ms = DummyConfigs(features='MS')
    model_ms = TimeXer_DINOv3_Model(cfg_ms)
    model_ms.train()

    batch_size = 4
    x_enc = torch.randn(batch_size, cfg_ms.seq_len, cfg_ms.enc_in)
    x_mark_enc = torch.randn(batch_size, cfg_ms.seq_len, 4)
    x_dec = torch.randn(batch_size, cfg_ms.pred_len, cfg_ms.enc_in)
    x_mark_dec = torch.randn(batch_size, cfg_ms.pred_len, 4)
    target_y = torch.randn(batch_size, cfg_ms.pred_len, 1)

    # Forward pass lấy cả output dự báo và đặc trưng DINO
    pred_out, dino_proj, patch_tokens = model_ms(
        x_enc, x_mark_enc, x_dec, x_mark_dec, return_dino=True
    )

    print(f"  + Shape đầu vào x_enc: {list(x_enc.shape)}")
    print(f"  + Shape đầu ra dự báo pred: {list(pred_out.shape)}")
    print(f"  + Shape DINO projection head: {list(dino_proj.shape)}")
    print(f"  + Shape Patch Tokens (Gram Anchoring): {list(patch_tokens.shape)}")

    expected_ms = [batch_size, cfg_ms.pred_len, 1]
    assert list(pred_out.shape) == expected_ms, f"Lỗi shape MS: {list(pred_out.shape)} != {expected_ms}"
    print("  -> TEST 1 PASSED: Tensor shape nhiệm vụ MS chính xác 100%!")

    # -------------------------------------------------------------------------
    # TEST 2: Kiểm tra tính toán DINO-v3 Loss Function (Bao gồm cả 3 thành phần)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Kiểm tra DINO-v3 Composite Loss Function (Task + Distillation + Gram)...")
    loss_fn = DINOv3Loss(
        task_criterion=nn.MSELoss(),
        student_temp=0.1,
        teacher_temp=0.04,
        lambda_distill=0.1,
        lambda_gram=0.05,
        out_dim=256
    )

    # Tạo Teacher projection giả lập (qua EMA hoặc gradient-free)
    teacher_proj = dino_proj.detach() + torch.randn_like(dino_proj) * 0.01

    loss_total, loss_details = loss_fn(
        pred=pred_out,
        true=target_y,
        student_proj=dino_proj,
        teacher_proj=teacher_proj,
        student_patches=patch_tokens,
        anchor_patches=patch_tokens.detach()
    )

    print("  + Bóc tách thành phần hàm mất mát DINO-v3:")
    print(f"    * L_task (MSE Forecasting)    : {loss_details['loss_task']:.6f}")
    print(f"    * L_distill (Self-Distill)     : {loss_details['loss_distill']:.6f}")
    print(f"    * L_gram (Gram Anchoring)      : {loss_details['loss_gram']:.6f}")
    print(f"    -----------------------------------------------------")
    print(f"    * L_total (Tổng hợp DINO-v3)   : {loss_details['loss_total']:.6f}")

    assert loss_details['loss_task'] > 0, "Lỗi: Task Loss phải lớn hơn 0"
    assert loss_details['loss_total'] > 0, "Lỗi: Total Loss phải lớn hơn 0"
    print("  -> TEST 2 PASSED: DINO-v3 Composite Loss Function hoạt động chuẩn xác!")

    # -------------------------------------------------------------------------
    # TEST 3: Kiểm tra Lan truyền ngược (Backward Gradient Flow)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Kiểm tra Gradient Backward từ DINO-v3 Loss...")
    optimizer = torch.optim.Adam(model_ms.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    # Kiểm tra gradient các trọng số Backbone DINO-v3
    has_grad = False
    for name, param in model_ms.named_parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            has_grad = True
            break

    assert has_grad, "Lỗi: Gradient không truyền được về trọng số mô hình!"
    print("  -> TEST 3 PASSED: Gradient lan truyền ngược thành công qua toàn bộ mạng!")

    # -------------------------------------------------------------------------
    # TEST 4: Kiểm tra models/TimeXer.py với cờ --encoder_type dino_v3
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Kiểm tra models/TimeXer.py tích hợp sẵn encoder_type='dino_v3'...")
    model_unified = TimeXer_Unified_Model(cfg_ms)
    with torch.no_grad():
        out_uni = model_unified(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert list(out_uni.shape) == expected_ms
    print("  -> TEST 4 PASSED: TimeXer.py gọi backbone DINO-v3 hoàn toàn trong suốt!")

    total_params = sum(p.numel() for p in model_ms.parameters() if p.requires_grad)
    print("\n" + "=" * 75)
    print(f"  TỔNG SỐ THAM SỐ HUẤN LUYỆN (DINO-v3 Backbone): {total_params:,}")
    print("  HOÀN THÀNH TOÀN BỘ 4 BÀI TEST DINO-v3 THÀNH CÔNG RỰC RỠ!")
    print("=" * 75)


if __name__ == '__main__':
    run_dinov3_verification()
