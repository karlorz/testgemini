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
                f"Create a friendly, cartoon-style 3D rendering of {prompt} in a cute and safe style"
            )

            print(f"提示詞: {enhanced_prompt}")

            response = self.model.generate_content(
                enhanced_prompt,
                generation_config={
                    'temperature': 0.9,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                stream=False
            )

            print(f"回應狀態: {getattr(response, 'prompt_feedback', 'No feedback')}")
            print(f"完整回應物件: {response}")
            print(f"回應屬性: {dir(response)}")
            
            # 檢查回應
            print(f"候選項數量: {len(response.candidates) if response.candidates else 0}")
            if not response.candidates or len(response.candidates) == 0:
                raise Exception("未收到任何回應")

            for candidate in response.candidates:
                print(f"候選項內容: {candidate}")
                print(f"候選項屬性: {dir(candidate)}")
                
                if not candidate.content:
                    print("候選項沒有內容")
                    continue
                
                print(f"Content 物件: {candidate.content}")
                print(f"Content 屬性: {dir(candidate.content)}")
                
                for part in candidate.content.parts:
                    print(f"Part 物件: {part}")
                    print(f"Part 屬性: {dir(part)}")
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
