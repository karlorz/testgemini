import google.generativeai as genai
from PIL import Image
from io import BytesIO
import json

class GeminiImageGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')

    def generate_image(self, prompt):
        try:
            generation_config = {
                "temperature": 0.4,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            # 確保提示為英文
            prompt = f"Generate an image of: {prompt}"
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False
            )
            
            if not response.candidates:
                raise Exception("No candidates in response")
                
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data
                    
            raise Exception("No image data in response")
            
        except Exception as e:
            print(f"詳細錯誤: {str(e)}")  # 記錄詳細錯誤
            raise Exception(f"圖像生成錯誤: {str(e)}")
