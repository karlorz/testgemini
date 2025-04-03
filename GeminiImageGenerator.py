from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

class GeminiImageGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate_image(self, prompt):
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            
            print(f"提示詞: {prompt}")
            print(f"回應狀態: {response.prompt_feedback}")
            
            if not response.candidates:
                raise Exception("未收到任何回應")
            
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(f"模型回應文字: {part.text}")
                elif part.inline_data is not None:
                    return part.inline_data.data
            
            raise Exception("回應中沒有圖像資料")
            
        except Exception as e:
            error_msg = f"詳細錯誤: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
