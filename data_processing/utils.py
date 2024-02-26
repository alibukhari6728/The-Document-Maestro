import base64
import io
import os
import re
from io import BytesIO
from typing import List, Tuple
from PIL import Image

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from pdf2image import convert_from_path


def looks_like_base64(sb: str) -> bool:
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data: str) -> bool:
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

# need to improve function name here 
def get_base64_image(base64_str: str) -> Image.Image:
    # Decode the base64 string
    image_data = base64.b64decode(base64_str)

    # Convert bytes to an image
    image = Image.open(BytesIO(image_data))

    return image

def resize_base64_image(base64_string: str, size: Tuple[int, int] = (128, 128)) -> str:
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Categorize elements by type
def categorize_elements(raw_pdf_elements: List[str]) -> Tuple[List[str], List[str]]:
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return texts, tables


def tokenize_text(texts: List[str]) -> List[str]:
    # Optional: Enforce a specific token size for texts
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )

    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)

    return texts_4k_token


def encode_image(image_path: str) -> str:
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_pdf_pages_as_images(pdf_path: str, output_folder: str) -> None:
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF to images
    pages = convert_from_path(pdf_path)

    # Save each page as an image
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f'page_{i+1}.jpg')
        page.save(image_path, 'JPEG')
        print(f'Saved page {i+1} as {image_path}')

def image_summarize(img_base64: str, prompt: str, model) -> str:
    """Make image summary"""

    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )

    return msg.content

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}
