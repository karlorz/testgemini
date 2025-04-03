import google.generativeai as genai
from google.generativeai import types
from io import BytesIO
import base64
import json

class GeminiImageGenerator:
    def __init__(self, api_key):
        # Configure the client with API key
        genai.configure(api_key=api_key)
        # Create a client instance
        self.client = genai.Client()
        
    def generate_image(self, prompt):
        try:
            print(f"提示詞: {prompt}")
            
            # Format the prompt for clarity
            formatted_prompt = f"Generate a detailed image of: {prompt}"
            
            # Call the API using the format from the documentation
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["Text", "Image"]
                )
            )
            
            print(f"Response type: {type(response)}")
            print(f"Response dir: {dir(response)}")
            
            # Extract the image data from the response parts
            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        print("找到圖像資料")
                        return base64.b64decode(part.inline_data.data)
                        
            # Alternative way to extract image data if the structure is different
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                print("找到圖像資料")
                                return base64.b64decode(part.inline_data.data)
            
            raise Exception("回應中沒有圖像資料")

        except Exception as e:
            error_msg = f"圖像生成錯誤: {str(e)}"
            print(error_msg)
            
            # Print full exception for debugging
            import traceback
            print(traceback.format_exc())
            
            raise Exception(error_msg)
