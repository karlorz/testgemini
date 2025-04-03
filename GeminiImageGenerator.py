import google.generativeai as genai
from google.generativeai.types import GenerationConfig

class GeminiImageGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')

    def generate_image(self, prompt):
        try:
            response = self.model.generate_content(
                contents=prompt,
                config=GenerationConfig(response_modalities=['Text', 'Image'])
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
            return None
        except Exception as e:
            raise Exception(f"圖像生成錯誤: {e}")
