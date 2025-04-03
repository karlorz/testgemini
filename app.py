from flask import Flask, request, render_template, send_file
from GeminiImageGenerator import GeminiImageGenerator
import os
from io import BytesIO

app = Flask(__name__)

# 確保 API 金鑰已設定為環境變數
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("必須設定 GOOGLE_API_KEY 環境變數。")

image_generator = GeminiImageGenerator(api_key=GOOGLE_API_KEY)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        try:
            image_bytes = image_generator.generate_image(prompt)
            if image_bytes:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                return render_template('index.html', image_data=base64_image)
            else:
                return render_template('index.html', error='圖像生成失敗。')
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

import base64

if __name__ == '__main__':
    app.run(debug=True)
