import pathlib
import google.generativeai as genai
from io import BytesIO
import base64
import json

class GeminiImageGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Use gemini-1.5-pro which has better support for image generation
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def generate_image(self, prompt):
        try:
            print(f"提示詞: {prompt}")
            
            # Format the prompt specifically for image generation using tools
            formatted_prompt = {
                "text": f"Generate a high-resolution, detailed image of: {prompt}. Make sure the image is vivid and clear."
            }
            
            # Set up the tool for image generation
            tools = [
                {
                    "function_declarations": [
                        {
                            "name": "generate_image",
                            "description": "Generate an image based on the input prompt",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "prompt": {
                                        "type": "string",
                                        "description": "The prompt to generate an image for"
                                    }
                                },
                                "required": ["prompt"]
                            }
                        }
                    ]
                }
            ]
            
            # Call the API with properly formatted request
            response = self.model.generate_content(
                formatted_prompt,
                generation_config={
                    "temperature": 0.9,
                    "top_p": 1,
                    "top_k": 32,
                },
                tools=tools,
                tool_config={"function_calling_config": {"mode": "auto"}},
                stream=False
            )
            
            # Detailed logging to help debug
            print(f"Response type: {type(response)}")
            print(f"Response dir: {dir(response)}")
            
            # Try to find any function calls that might contain image data
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    print(f"Candidate: {candidate}")
                    
                    # Check for function calls in the response
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            # Look for function calls
                            if hasattr(part, 'function_call') and part.function_call:
                                print(f"Found function call: {part.function_call}")
                                # Attempt to extract image data
                                if part.function_call.name == "generate_image":
                                    # Extract the prompt from MapComposite object directly
                                    # Access the fields directly instead of using json.loads
                                    try:
                                        args = part.function_call.args
                                        # MapComposite objects can be accessed like dictionaries
                                        image_prompt = args.get("prompt", prompt)
                                        if hasattr(image_prompt, "string_value"):  # It might be wrapped in a Value object
                                            image_prompt = image_prompt.string_value
                                        
                                        print(f"Extracted image prompt: {image_prompt}")
                                        
                                        # Call the model directly with the extracted prompt
                                        imagen_response = self.model.generate_content(
                                            f"Please generate an image of: {image_prompt}. Return it as an inline image only, without any text.",
                                            stream=False
                                        )
                                        
                                        # Try to extract image from the imagen response
                                        if hasattr(imagen_response, 'parts'):
                                            for img_part in imagen_response.parts:
                                                if hasattr(img_part, 'inline_data') and img_part.inline_data:
                                                    print("Found image data")
                                                    return base64.b64decode(img_part.inline_data.data)
                                    except Exception as e:
                                        print(f"Error extracting prompt from function call: {e}")
                            
                            # Direct check for inline data
                            if hasattr(part, 'inline_data') and part.inline_data:
                                print("Found direct image data")
                                return base64.b64decode(part.inline_data.data)
            
            # If we got here, we need to try a different approach
            print("First method failed, trying direct image generation...")
            
            # Try a more direct approach with simple prompt
            direct_response = self.model.generate_content(
                f"Generate an image of {prompt}. Output only the image with no text.",
                stream=False
            )
            
            # Check for image data in the direct response
            if hasattr(direct_response, 'parts'):
                for part in direct_response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        print("Found image in direct response")
                        return base64.b64decode(part.inline_data.data)
                        
            raise Exception("No image data found in any of the responses")

        except Exception as e:
            error_msg = f"圖像生成錯誤: {str(e)}"
            print(error_msg)
            
            # Print full exception for debugging
            import traceback
            print(traceback.format_exc())
            
            raise Exception(error_msg)
