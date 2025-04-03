import pathlib
import google.generativeai as genai
from io import BytesIO
import base64

class GeminiImageGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Configure the model with the correct model name for image generation
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
        
    def generate_image(self, prompt):
        try:
            print(f"提示詞: {prompt}")
            
            # Call the image generation API with correct parameters for Gemini 2.0
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "responseModalities": ["Text", "Image"],
                    "temperature": 1.0,
                },
                stream=False
            )
            
            print(f"回應狀態: {getattr(response, 'prompt_feedback', 'No feedback')}")
            
            # Extract the image data from the response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            print("找到圖像資料")
                            # Return the raw bytes from the inline_data
                            return base64.b64decode(part.inline_data.data)
            
            # Alternative check if the response structure is different
            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        print("找到圖像資料")
                        return base64.b64decode(part.inline_data.data)
            
            raise Exception("回應中沒有圖像資料")

        except Exception as e:
            error_msg = f"圖像生成錯誤: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
