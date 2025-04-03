from google import genai
from google.genai import types

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
            
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
            return None
        except Exception as e:
            raise Exception(f"圖像生成錯誤: {e}")
