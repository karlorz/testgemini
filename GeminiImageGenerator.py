import google.generativeai as genai
from io import BytesIO
import base64
import traceback

class GeminiImageGenerator:
    def __init__(self, api_key):
        # Configure the API with your key
        genai.configure(api_key=api_key)
        # Using the model name directly - no client instance needed
        self.model_name = "gemini-2.0-flash-exp-image-generation"
        
    def generate_image(self, prompt):
        try:
            print(f"提示詞: {prompt}")
            
            # Format the prompt for clarity
            formatted_prompt = f"Generate a detailed image of: {prompt}"
            
            # Call the API using the direct method
            response = genai.generate_content(
                model=self.model_name,
                contents=formatted_prompt,
                generation_config={
                    "response_mime_types": ["image/png"],
                }
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
            print(traceback.format_exc())
            
            raise Exception(error_msg)
