# The Document Maestro
A langchain-based retrieval augmented generated pipeline for multi-modal PDF analysis with a focus on ESG document analysis.

![Screenshot from 2024-02-26 16-28-24](https://github.com/alibukhari6728/The-Document-Maestro/assets/63595396/1f8d9f3f-0ba3-47b1-bdfd-92a0c8900fdc)

## Environment

To set-up the environment, please run the following commands:

```
- conda env create -f environment.yml
- conda activate pdfRAG
```

in case the above environment set-up does not work for you, please try the following commands:

```
- conda create -n "pdfRAG" python==3.10
- conda activate pdfRAG
- pip install -U langchain openai chromadb langchain-experimental
- pip install "unstructured[all-docs]" pillow pydantic lxml pillow matplotlib chromadb tiktoken
- pip intall streamlit
```

#### API-Key

Please add your OPENAI API-key with:

- `export OPENAI_API_KEY= <your-api-key-here>`

## Launch APP

To run the web-app, please run:

- `streamlit run app.py`

