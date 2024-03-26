# The Document Maestro
A Langchain-powered retrieval-augmented-generation pipeline for comprehensive multi-modal analysis of PDFs, specifically tailored for ESG document probing.

![Screenshot from 2024-02-26 16-28-24](https://github.com/alibukhari6728/The-Document-Maestro/assets/63595396/1f8d9f3f-0ba3-47b1-bdfd-92a0c8900fdc)

## Environment

To weave the environment for this digital alchemy, follow these incantations:

```
conda env create -f environment.yml
conda activate pdfRAG
```

If the above does not work for you, fear not. Try these alternative spells:

```
conda create -n "pdfRAG-env" python==3.10
conda activate pdfRAG-env
pip install -U langchain openai chromadb langchain-experimental
pip install "unstructured[all-docs]" pillow pydantic lxml pillow matplotlib chromadb tiktoken
pip intall streamlit
```

#### API-Key

Whisper your OPENAI API-key:

- `export OPENAI_API_KEY=<your-api-key-here>`

## Launch APP

To set sail, chant:

- `streamlit run app.py`

# Upcoming Updates:

- Knowledge Graphs (very soon)
- Corrective Strategy (supervising LLM)
- RPN-based chunk optimization
- Reciprocal Reranking
- Multimodal OpenLLM based local engine
