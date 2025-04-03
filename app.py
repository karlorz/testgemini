from flask import Flask, request, render_template, send_file
from GeminiImageGenerator import GeminiImageGenerator
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()  # 載入 .env 檔案

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
            # 記錄請求
            print(f"接收到的提示: {prompt}")
            
            image_bytes = image_generator.generate_image(prompt)
            if not image_bytes:
                raise Exception("未能生成圖像")
            
            # 記錄成功
            print("圖像生成成功")
            
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return render_template('index.html', 
                                image_data=base64_image,
                                prompt=prompt,
                                success_message="圖像生成成功！")
                                
        except Exception as e:
            # 記錄錯誤
            print(f"錯誤詳情: {str(e)}")
            return render_template('index.html', 
                                error=f"圖像生成錯誤: {str(e)}",
                                prompt=prompt)
    return render_template('index.html')

import base64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
