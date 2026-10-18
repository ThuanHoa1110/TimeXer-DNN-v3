import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


# =======================================================================================
# [YÊU CẦU 1] KIẾN TRÚC MẠNG DNN-V3 (DEEP NEURAL NETWORK VERSION 3) THAY THẾ ENCODER BACKBONE
# =======================================================================================

class DNN_v3_Layer(nn.Module):
    """
    DNN-v3 Layer: Khối nơ-ron sâu 3 giai đoạn (3-Stage Deep Neural Network Block)
    được thiết kế chuyên biệt để thay thế Encoder Attention của TimeXer:
      - Giai đoạn 1 (Stage 1): Intra-patch Temporal Feature Representation MLP
      - Giai đoạn 2 (Stage 2): Cross-Variate Exogenous Fusion MLP (Tích hợp thông tin ngoại sinh)
      - Giai đoạn 3 (Stage 3): Deep Feed-Forward Expansion & Residual Projection MLP
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(DNN_v3_Layer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # --- Giai đoạn 1: Biểu diễn đặc trưng nội sinh qua mạng Dense sâu (MLP 1) ---
        self.stage1_dense1 = nn.Linear(d_model, d_ff)
        self.stage1_dense2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # --- Giai đoạn 2: Tích hợp thông tin biến ngoại sinh qua Cross-MLP (MLP 2) ---
        # Thay thế cơ chế Cross-Attention bằng mạng Dense phi tuyến kết hợp cơ chế Gated/Residual
        self.ex_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cross_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # --- Giai đoạn 3: Mở rộng không gian phi tuyến FFN và Residual (MLP 3) ---
        self.stage3_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        x: [B * n_vars, patch_num + 1, d_model] - Biểu diễn các patch nội sinh + global token
        cross: [B, L_ex, d_model] - Biểu diễn biến ngoại sinh (exogenous)
        """
        B, L_ex, D = cross.shape
        total_tokens = x.shape[0]
        n_vars = total_tokens // B

        # --- STAGE 1: Feature Extraction via Deep Dense Layer ---
        res1 = x
        x_dense = self.dropout(self.activation(self.stage1_dense1(x)))
        x_dense = self.dropout(self.stage1_dense2(x_dense))
        x = self.norm1(res1 + x_dense)

        # --- STAGE 2: Exogenous Information Fusion (Cross-MLP) ---
        # Tổng hợp đặc trưng biến ngoại sinh trên toàn bộ variates
        ex_summary = torch.mean(cross, dim=1, keepdim=True)  # [B, 1, D]
        ex_summary = self.ex_proj(ex_summary)                # [B, 1, D]
        if n_vars > 1:
            ex_summary = ex_summary.repeat_interleave(n_vars, dim=0)  # [B * n_vars, 1, D]

        # Tách global token của chuỗi nội sinh (token cuối cùng ở vị trí -1)
        x_glb_ori = x[:, -1:, :]  # [B * n_vars, 1, D]

        # Kết hợp nội sinh & ngoại sinh qua mạng nơ-ron sâu
        fused_input = torch.cat([x_glb_ori, ex_summary], dim=-1)  # [B * n_vars, 1, 2*D]
        glb_fused = self.dropout(self.cross_fusion(fused_input))  # [B * n_vars, 1, D]
        x_glb = self.norm2(x_glb_ori + glb_fused)

        # Ghép lại chuỗi: [các patch nội sinh, global token đã được cập nhật ngoại sinh]
        x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # --- STAGE 3: Feedforward Expansion & Final Residual Normalization ---
        y = self.stage3_ffn(x)
        return self.norm3(x + y)


class DNN_v3_Encoder(nn.Module):
    """
    DNN-v3 Encoder Backbone:
    Xếp chồng e_layers khối DNN-v3 Layers kèm lớp chuẩn hóa LayerNorm cuối.
    Hoàn toàn tương thích và thay thế cho Transformer Encoder gốc của TimeXer.
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(DNN_v3_Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


# =======================================================================================
# MÔ HÌNH CHÍNH TIMEXER ĐÃ THAY THẾ BACKBONE BẰNG DNN-V3
# =======================================================================================

class Model(nn.Module):
    """
    TimeXer với Encoder/Backbone được thay thế bằng mạng DNN-v3.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        # 1. Embedding Layers
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # 2. [CẬP NHẬT YÊU CẦU 1] Backbone Encoder DNN-v3
        # Thay thế Transformer Attention Encoder bằng DNN_v3_Encoder
        self.encoder = DNN_v3_Encoder(
            [
                DNN_v3_Layer(
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 3. Projection Head
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(
            configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        # Chạy qua DNN-v3 Encoder
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        # Chạy qua DNN-v3 Encoder
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
        else:
            return None
