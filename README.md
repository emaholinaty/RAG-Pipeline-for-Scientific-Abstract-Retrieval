# RAG-Pipeline-for-Scientific-Abstract-Retrieval (2026)
RAG system for querying scientific abstracts. The pipeline combines arXiv data ingestion, text chunking, embeddings, FAISS retrieval, semantic reranking, and FLAN-T5-Large generation. Results show strong information retrieval capabilities but limitations in complex reasoning.

### Authors
* Ema Holinaty
* Esther Menéndez
* Guillermo García 

## OBJECTIVES

The main objective of this project is to build a structured Retrieval-Augmented Generation (RAG) pipeline capable of extracting accurate information from scientific abstracts. The selected topic for the system is **"open clusters AND Gaia"**. 

## PIPELINE
The system is divided into the following key stages:

1. Data Collection:
Scientific abstracts are retrieved from the arXiv API, cleaned, and stored in CSV format for further processing.

2. Chunking:
Documents are split into chunks of 800 characters with an overlap of 100 characters to preserve contextual information between segments.

3. Embedding Generation:
Text chunks are transformed into 768-dimensional vector representations using the **BAAI/bge-base-en-v1.5** embedding model.

5. Vector Indexing (FAISS)
An efficient vector search index is built using **HNSW (Hierarchical Navigable Small World)** and inner-product similarity, equivalent to cosine similarity after vector normalization.

5. Reranking
The top 5 retrieved documents are semantically refined using a **MiniLM Cross-Encoder** reranking model.

6. Answer Generation
Final responses are generated using **Google FLAN-T5-Large (770M parameters)**, selected for its strong performance and efficiency in resource-constrained environments such as Google Colab.

## MAIN LIBRARIES

* **arxiv** – Access to the arXiv API.
* **langchain-text-splitters** and **langchain-core** – Text processing and prompt management.
* **sentence-transformers** – Embedding and reranking models.
* **faiss-cpu** – Vector search engine.
* **transformers** – Loading and running the language model.
* **pandas** and **numpy** – Data manipulation and numerical operations.

## EVALUATION

- UMAP Visualization: UMAP dimensionality reduction is used to visually verify that retrieved chunks are semantically close to the user's query.

- Robustness: The system includes explicit instructions to return **"Not enough information"** when queries are irrelevant or unintelligible, reducing hallucinations.

- Performance: The chunking and embedding stages require approximately **8 minutes**, while individual answer generation takes around **1 minute**.

## LIMITATIONS

* The model tends to perform extractive summarization and literal information retrieval rather than deep reasoning about complex physical concepts.
* FLAN-T5 has a maximum input length of **512 tokens**, limiting the total context that can be processed at once.
