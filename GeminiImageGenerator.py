from google import genai
from google.genai import types
from io import BytesIO
import base64
import traceback

class GeminiImageGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        
    def generate_image(self, prompt):
        try:
            print(f"提示詞: {prompt}")
            
            # Keep the prompt as is without additional formatting
            # Call the API using the verified method from test.py
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            
            print("Response received")
            
            # Extract image data using the verified structure from test.py
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    print("找到圖像資料")
                    # 直接返回原始數據
                    return part.inline_data.data
            
            raise Exception("回應中沒有圖像資料")

        except Exception as e:
            error_msg = f"圖像生成錯誤: {str(e)}"
            print(error_msg)
            
            # Print full exception for debugging
            print(traceback.format_exc())
            
            raise Exception(error_msg)
