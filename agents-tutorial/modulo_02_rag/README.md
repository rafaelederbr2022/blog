# Module 2: RAG (Retrieval-Augmented Generation)

## Description

This module covers Retrieval-Augmented Generation techniques from basic to advanced. You will learn how to build document-based search and answer generation systems, including multi-document retrieval, metadata filtering, query optimization, hybrid search, reranking, and citation-based responses.

## Exercises

| # | Exercise | Description |
|---|----------|-------------|
| 06 | Basic RAG | Document loading, splitting, embedding, and querying with ChromaDB |
| 07 | Multi-document RAG | Searching across multiple data sources with result merging |
| 08 | Metadata Filtering | Filtering indexed documents by metadata attributes |
| 09 | Query Expansion | Generating query variations to improve recall |
| 10 | Query Decomposition | Breaking complex questions into sub-questions |
| 11 | Hybrid Search | Combining BM25 with embeddings for better retrieval |
| 12 | Reranking | Reordering results with cross-encoder models |
| 13 | Context Compression | Compressing retrieved context before sending to the LLM |
| 14 | Citation-based RAG | Generating answers with source citations |

## Prerequisites

- Module 1 (LangChain Fundamentals) completed
- OpenAI API key configured
- ChromaDB installed (included in requirements.txt)

## Key Concepts

- Document loaders and text splitters
- Embeddings (OpenAI text-embedding-3-small)
- Vector stores (ChromaDB)
- Similarity search and MMR
- BM25 keyword search
- Cross-encoder reranking
- Context compression techniques
- Source attribution and citations
