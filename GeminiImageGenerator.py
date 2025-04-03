import pathlib
import google.generativeai as genai
from PIL import Image
from io import BytesIO

class GeminiImageGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')

    def generate_image(self, prompt):
        try:
            enhanced_prompt = (
                f"Hi, can you create a 3d rendered image of {prompt}?"
            )

            print(f"提示詞: {enhanced_prompt}")

            response = self.model.generate_content(
                enhanced_prompt,
                generation_config={
                    'temperature': 0.9,
                },
                tools=[{"type": "IMAGE_GENERATION"}],
                stream=False
            )

            print(f"回應狀態: {getattr(response, 'prompt_feedback', 'No feedback')}")

            # 檢查回應
            if not response.candidates or len(response.candidates) == 0:
                raise Exception("未收到任何回應")

            for candidate in response.candidates:
                if not candidate.content:
                    continue
                
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"文字回應: {part.text}")
                    if hasattr(part, 'inline_data') and part.inline_data:
                        print("找到圖像資料")
                        return part.inline_data.data


            raise Exception("回應中沒有圖像資料")

        except Exception as e:
            error_msg = f"圖像生成錯誤: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
