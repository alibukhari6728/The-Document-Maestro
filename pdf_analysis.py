from PIL import Image
from typing import Optional, Tuple

from data_processing.multimodal_data import MultimodalESGData
from data_processing.utils import get_base64_image, is_image_data, looks_like_base64

from rag.retriever_builder import create_multi_vector_retriever
from rag.chain_builder import multi_modal_rag_chain

def pdf_analyzer(pdf_path: str, query: str) -> Tuple[str, Optional[Image.Image]]:

    #Extract data from pdf file
    esg_document = MultimodalESGData(pdf_path)

    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(esg_document)

    # Create RAG chain
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

    # Retrieve Relevant documents
    docs = retriever_multi_vector_img.get_relevant_documents(query, limit=6)

    # Ask the final LLM
    invoked_response = chain_multimodal_rag.invoke(query)

    # Check if retrieved image makes sense
    returned_image = None
    for doc in docs:
        if looks_like_base64(doc) and is_image_data(doc):
            returned_image = get_base64_image(doc)
            break

    return invoked_response, returned_image

