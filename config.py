# environment

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

###########################################################################

# models

from langchain_community.chat_models import ChatOpenAI

text_embedding_model = ChatOpenAI(temperature=0, model="gpt-4")
image_embedding_model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

multimodal_LLM = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)