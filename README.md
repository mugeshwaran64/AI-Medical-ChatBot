# AI Medical Chatbot

Minimal RAG-style medical chatbot using local HuggingFace embeddings, Pinecone vector store, and Groq (GROQ) LLM via LangChain adapters.

## Requirements
- Python 3.10+
- Conda or virtualenv (recommended)
- Pinecone account & index
- GROQ API key (or OPENAI API key if using OpenAI)
- Packages in requirements.txt (or install below)

Install:
```bash
pip install -r requirements.txt
```

## Setup
1. Copy `.env.example` to `.env` (or create `.env`) in project root.
2. Add keys to `.env`:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment   # e.g. us-east1-gcp
GROQ_API_KEY=your_groq_api_key                   # fallback for OPENAI_API_KEY
OPENAI_API_KEY=your_openai_api_key               # optional
```
The app uses OPENAI_API_KEY or falls back to GROQ_API_KEY (set GROQ_API_KEY when switching to Groq).

3. Ensure Pinecone index is created and schema is set (see below).

## Usage
Run `app.py`:
```bash
python app.py
```

Interact with the chatbot at `http://localhost:5001` (or your configured host:port).

## Indexing Documents
To index documents, use `index.py`:
```bash
python index.py
```

Ensure your documents are in the `data/` directory (subdirectories okay).

## Schema
The index schema should have the following fields:
- `text`: the document text (string)
- `metadata`: any metadata (JSON object)
- `embedding`: the text embedding (vector)

## License
MIT License. See `LICENSE` for details.

## Acknowledgments
- [LangChain](https://langchain.com)
- [Pinecone](https://www.pinecone.io)
- [Hugging Face](https://huggingface.co)
- [Groq](https://groq.com)

938998269993.dkr.ecr.eu-north-1.amazonaws.com/medical-chatbot

today pushed completed