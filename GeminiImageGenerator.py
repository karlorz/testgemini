import pathlib
import google.generativeai as genai
from PIL import Image
from io import BytesIO

class GeminiImageGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-pro-vision')

    def generate_image(self, prompt):
        try:
            enhanced_prompt = (
                f"Generate a high-quality, detailed image based on this description: {prompt}. "
                "Make it visually striking with vibrant colors."
            )

            print(f"提示詞: {enhanced_prompt}")

            response = self.model.generate_content(
                enhanced_prompt,
                stream=False
            )
            
            print(f"回應狀態: {response.prompt_feedback}")
            
            for part in response.text:
                print(f"模型回應: {part}")

            if not response.candidates:
                raise Exception("未收到任何回應")

            content = response.candidates[0].content
            
            if not content.parts:
                raise Exception("回應中沒有內容")

            for part in content.parts:
                if part.text:
                    print(f"文字回應: {part.text}")
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data

            raise Exception("回應中沒有圖像資料")

        except Exception as e:
            error_msg = f"圖像生成錯誤: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
