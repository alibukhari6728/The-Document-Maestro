import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from data_processing.multimodal_data import MultimodalESGData

def create_multi_vector_retriever(document_data :MultimodalESGData) -> MultiVectorRetriever:
    
    """
    Create a retriever that indexes summaries, but returns raw images or texts.

    Parameters:
        MultimodalESG data object

    Returns:
        MultiVectorRetriever: An instance of the MultiVectorRetriever class.    
    
    """

    # The vectorstore to use to index the summaries
    vectorstore = Chroma(
        collection_name="mm_rag_cj_blog", embedding_function=OpenAIEmbeddings()
    )

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        if doc_summaries:
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    add_documents(retriever, document_data.text_summaries, document_data.texts)
    add_documents(retriever, document_data.table_summaries, document_data.tables)
    add_documents(retriever, document_data.image_summaries, document_data.images)

    return retriever