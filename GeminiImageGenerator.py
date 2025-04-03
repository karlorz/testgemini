import google.generativeai as genai
from PIL import Image
from io import BytesIO

class GeminiImageGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def generate_image(self, prompt):
        try:
            # 使用 Imagen 模型 (Gemini 的圖像生成專用模型)
            response = genai.generate_images(
                model='imagen-3.0-generate-002',
                prompt=prompt,
                config={
                    'number_of_images': 1,
                    'aspect_ratio': '1:1',
                }
            )
            
            print(f"提示詞: {prompt}")
            
            if not response.generated_images:
                raise Exception("No images generated")
                
            image = response.generated_images[0]
            return image.image.image_bytes
            
        except Exception as e:
            error_msg = f"詳細錯誤: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
