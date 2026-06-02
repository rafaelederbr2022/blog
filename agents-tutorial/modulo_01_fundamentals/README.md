# Module 1: LangChain Fundamentals

## Description

This module covers the foundational concepts of LangChain, providing the essential building blocks for developing applications with Large Language Models. You will learn how to create dynamic prompts, call tools, maintain conversation context, compose processing pipelines, and stream responses in real-time.

## Exercises

| # | Exercise | Description |
|---|----------|-------------|
| 01 | Prompt Templates & Structured Output | Dynamic prompt templates with variables and schema validation using Pydantic |
| 02 | Tool Calling (Function Calling) | How LLMs invoke external tools and functions |
| 03 | Conversation with Memory | Maintaining context between conversation turns |
| 04 | Chains and Pipelines | Sequential composition of components using LCEL |
| 05 | Streaming Responses | Incremental token reception from the LLM |

## Prerequisites

- Python 3.11+ installed
- Virtual environment activated (`<YOUR_VENV_PATH>\Scripts\activate`)
- Dependencies installed (`pip install -r requirements.txt`)
- OpenAI API key configured in `.env`

## Key Concepts

- PromptTemplate and ChatPromptTemplate
- Structured Output with Pydantic validation
- Function Calling / Tool Calling
- ConversationBufferMemory and ConversationSummaryMemory
- LangChain Expression Language (LCEL)
- Streaming with `astream` and callbacks
- Chain composition (prompt → LLM → parser)
