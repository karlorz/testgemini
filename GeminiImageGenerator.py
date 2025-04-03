from google import genai
from google.generativeai import GenerativeModel
from PIL import Image
from io import BytesIO

class GeminiImageGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = GenerativeModel('gemini-2.0-flash-exp-image-generation')

    def generate_image(self, prompt):
        try:
            response = self.model.generate_content(
                contents=[{'text': prompt}],
                generation_config={'temperature': 0.9},
                stream=False
            )
            
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data
            return None
        except Exception as e:
            raise Exception(f"圖像生成錯誤: {e}")
