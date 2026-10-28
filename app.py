"""
TimeXer-DINOv3 Weather AI - Web Server & Production Model Serving
Học phần: Chuyên đề Trí tuệ nhân tạo và Khoa học dữ liệu (Lớp 65TTNT)
Hỗ trợ triển khai hệ thống (Web API & UI) tích hợp mô hình TimeXer-DINOv3 thật
"""

import os
import sys
import json
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import pandas as pd

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from inference_engine import TimeXerInferenceEngine, TORCH_AVAILABLE

PORT = int(os.environ.get("PORT", 8080))

# Khởi tạo engine toàn cục
engine = TimeXerInferenceEngine()


class TimeXerAPIHandler(SimpleHTTPRequestHandler):
    """Handler phục vụ cả giao diện Web tĩnh và REST API cho Model Inference"""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # Endpoint kiểm tra trạng thái Model Serving
        if parsed.path == "/api/model-info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            info = {
                "model_name": "TimeXer_DINOv3",
                "backbone": "DINO-v3 Multi-Head Cross Attention with LayerScale",
                "loss_function": "DINO-v3 Composite Loss (MSE + Distill + Gram)",
                "torch_available": TORCH_AVAILABLE,
                "device": str(engine.device),
                "checkpoint_loaded": os.path.exists(engine.checkpoint_path),
                "features": engine.feature_cols
            }
            self.wfile.write(json.dumps(info, ensure_ascii=False).encode('utf-8'))
            return

        # Phục vụ file giao diện mặc định
        if parsed.path in ["", "/"]:
            self.path = "/index.html"
            return super().do_GET()

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        # API Suy luận mô hình thực tế: /api/predict
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            
            try:
                payload = json.loads(body)
            except Exception as e:
                payload = {}

            horizon = int(payload.get('horizon', 24))
            use_exogenous = bool(payload.get('use_exogenous', True))
            station_id = payload.get('station', 'st01_hoankiem')

            # Nếu người dùng gửi mảng dữ liệu custom
            custom_data = payload.get('data', None)
            if custom_data and len(custom_data) >= 96:
                input_df = pd.DataFrame(custom_data)
            else:
                # Nạp từ file dữ liệu trạm thực tế
                st_file = os.path.join(BASE_DIR, "data", "stations_sample", f"{station_id}.csv")
                if not os.path.exists(st_file):
                    st_file = os.path.join(BASE_DIR, "data", "private", "real_weather_large_dataset.csv")

                if os.path.exists(st_file):
                    df_full = pd.read_csv(st_file)
                    input_df = df_full.iloc[-96:].copy()
                else:
                    # Fallback tạo 96 mốc chuẩn
                    dates = pd.date_range(end=pd.Timestamp.now(), periods=96, freq='5min')
                    input_df = pd.DataFrame({
                        'timestamp': dates,
                        'temperature': 28.0 + np.sin(np.arange(96) * 0.1) * 3.5,
                        'solar_radiation': np.clip(np.sin(np.arange(96) * 0.1) * 600, 0, 1000),
                        'wind_speed_10m': 2.5 + np.random.normal(0, 0.4, 96),
                        'humidity': 75.0 - np.sin(np.arange(96) * 0.1) * 10,
                        'surface_pressure': 1013.0
                    })

            # Thực thi suy luận PyTorch
            try:
                result = engine.predict(
                    df_input=input_df,
                    horizon=horizon,
                    use_exogenous=use_exogenous
                )

                response_payload = {
                    "status": "success",
                    "model": "TimeXer_DINOv3",
                    "lookback_history": result['lookback_history'][['timestamp', 'temperature']].to_dict(orient='records'),
                    "predictions": result['predictions'].to_dict(orient='records'),
                    "metrics": result['metrics'],
                    "exogenous_attribution": result['exogenous_attribution']
                }
            except Exception as e:
                response_payload = {
                    "status": "error",
                    "message": str(e)
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload, default=str, ensure_ascii=False).encode('utf-8'))
            return

        self.send_error(404, "Endpoint not found")


def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, TimeXerAPIHandler)
    print("=" * 75)
    print(f"🚀 TimeXer-DINOv3 Model Serving API & Web Server khởi động tại:")
    print(f"👉 http://localhost:{PORT}")
    print(f"[*] Endpoint suy luận: POST http://localhost:{PORT}/api/predict")
    print(f"[*] Endpoint kiểm tra: GET  http://localhost:{PORT}/api/model-info")
    print("=" * 75)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Dừng server.")
        httpd.server_close()


if __name__ == '__main__':
    run_server()
