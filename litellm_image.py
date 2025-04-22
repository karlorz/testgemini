import base64
import os
import openai
client = openai.OpenAI(
    api_key="",
    base_url="" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)


response = client.chat.completions.create(
    model="gemini/gemini-2.0-flash-exp-image-generation",
    messages=[{"role": "user", "content": "Generate an image of a cat"}],
    modalities=["image", "text"],
)
assert response.choices[0].message.content is not None # "data:image/png;base64,e4rr.."
image_base64_content = response.choices[0].message.content.split(",")[1]
image_bytes = base64.b64decode(image_base64_content)
with open("cat_image.png", "wb") as f:
    f.write(image_bytes)
