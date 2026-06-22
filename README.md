# Semantic Code Search Engine

An AI-powered semantic code search engine that allows developers to search a Python codebase using natural language queries.

## Features

* Semantic search using Sentence Transformers
* AST-based Python function extraction
* Multi-file codebase indexing
* Cosine similarity ranking
* Streamlit web interface
* Top-k code retrieval

## Tech Stack

* Python
* Sentence Transformers
* Scikit-learn
* Streamlit
* AST (Abstract Syntax Tree)

## Project Architecture

User Query
↓
Sentence Transformer
↓
Query Embedding
↓
Cosine Similarity Search
↓
Ranked Results
↓
Code Display

## Example Queries

* verify login credentials
* create database connection
* handle payment transaction

## Installation

```bash
git clone <your-repository-url>
cd semantic-code-search-engine

pip install -r requirements.txt
```

## Run the Application

Build the index:

```bash
python build_index.py
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

## Screenshots

Add screenshots here after uploading them to the repository.

## Future Improvements

* Upload custom codebases
* ZIP file support
* Better ranking metrics
* Support for additional programming languages

## Author

Built as an AI Engineering project using semantic search and embeddings.
 
