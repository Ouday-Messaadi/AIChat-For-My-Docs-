# 🧠 AIChat For My Docs

A powerful Retrieval-Augmented Generation (RAG) Python app that lets you load your own documents, search intelligently using hybrid/semantic/query expansion methods, and get accurate natural language answers powered by LLMs.

## 🚀 Features

- Load and split Markdown (`.md`) documents
- Store embeddings in a persistent **Chroma DB**
- Use **hybrid retrieval** (BM25 + vector search)
- Support for **semantic reranking**
- Simple **query expansion** logic (Word2Vec/embedding-based)
- Powered by 🤗 `sentence-transformers` + OpenRouter LLMs
- Interactive CLI to ask questions and receive smart answers

## 📁 Project Structure

.
├── aicfmd_v1.py # Main RAG system
├── data/ # Directory for your .md files
├── chroma/ # Persisted Chroma DB
├── .env # Contains your API key for OpenRouter
├── README.md # You're reading it

perl
Copy
Edit

## 🧰 Requirements

Make sure to install the dependencies:

```bash
pip install -r requirements.txt
Minimal requirements.txt:

langchain
langchain-community
langchain-huggingface
chromadb
openai
python-dotenv
sentence-transformers
```
🔑 .env Configuration
Create a .env file in the root folder:


DEEPSEEK_API_KEY=your_openrouter_api_key
You can get a free OpenRouter key from https://openrouter.ai/

📥 Adding Your Documents
Place your .md files inside the data/ folder.

Example:

kotlin

data/
└── alice_in_wonderland.md
🧪 Running the App
Run the script from terminal:

```bash

python aicfmd_v1.py
```
You'll see:

```bash

Creating semantic database...
🧠 Ask anything! (type 'exit' to quit)
📝 Your question:
```
Example Interaction
```vbnet

📝 Your question: What is Alice doing in the story?
--- Result 1 ---
Content Snippet: Alice was not a bit hurt, and she jumped up on to her feet in a moment...

💬 Answer: Alice is exploring Wonderland, following the White Rabbit and facing many surreal challenges...

📝 Your question: exit
👋 Exiting. Goodbye!`
```
⚙️ Search Modes
By default, the app uses "semantic_with_rerank" search. You can easily change this in aicfmd_v1.py:

```python

documents = rag_system.make_query(query=query, method="hybrid")
```
Supported methods:

semantic_with_rerank

hybrid

query_expansion

basic (raw vector search)

🧠 Future Ideas
Add PDF parsing with automatic Markdown conversion

Support chat history with memory

Web frontend using Streamlit or Flask

Plug in multilingual embeddings (e.g., LaBSE or multilingual MPNet)

👤 Author
Created by Ouday Messaadi
