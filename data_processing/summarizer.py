import os
from typing import List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from data_processing.utils import image_summarize, encode_image

def get_summarization_prompt(mode: str) -> str:

    if mode == "text" or "table":
        prompt = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """

    else:
        prompt = f"You are an assistant tasked with summarizing {mode} for retrieval. These summaries will be embedded and used to retrieve the raw image. Give a concise summary of the {mode} that is well optimized for retrieval."

    return prompt

# Generate summaries of text elements
def generate_text_summaries(texts: List[str], text_model) -> List[str]:
    """
    Summarize text elements
    texts: List of str
    """

    # Prompt
    prompt_text = get_summarization_prompt("text")

    prompt = ChatPromptTemplate.from_template(prompt_text)

    summarize_chain = {"element": lambda x: x} | prompt | text_model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []

    # Apply to text if texts are provided 
    if texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

    return text_summaries

def generate_table_summaries(tables: List[str], text_model) -> List[str]:
    """
    Summarize tables
    tables: List of str
    """

    # Prompt
    prompt_text = get_summarization_prompt("table")

    prompt = ChatPromptTemplate.from_template(prompt_text)

    summarize_chain = {"element": lambda x: x} | prompt | text_model | StrOutputParser()


    # Initialize empty summaries
    table_summaries = []

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return table_summaries


def generate_image_summaries(path: str, model) -> Tuple[List[str], List[str]]:
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = get_summarization_prompt("image")

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt, model))

    return img_base64_list, image_summaries