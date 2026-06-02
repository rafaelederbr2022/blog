# LangChain & LangGraph Study Project

A comprehensive, hands-on study project covering the LangChain ecosystem — including LangChain, LangGraph, LangFlow, LangSmith, MCP, ADK, DeepEval, and Context Engineering. The project is organized into 13 progressive modules with 94 practical exercises, culminating in a complete corporate agent as the final project.

## Prerequisites

- **Python 3.11+**
- **OpenAI API key** (required for all modules)
- Optional: LangSmith API key, Google API key, DeepEval API key, PostgreSQL, Redis

## Setup Instructions

1. **Activate the virtual environment:**

   ```bash
   C:\awsenv\venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   ```bash
   copy .env.example .env
   ```

   Open `.env` and fill in your API keys. At minimum, set `OPENAI_API_KEY`.

## Multi-Provider Configuration

The project supports multiple LLM providers. By default, OpenAI is used. You can switch to AWS Bedrock by setting the `LLM_PROVIDER` environment variable.

### Selecting a Provider

In your `.env` file, set:

```bash
# Use OpenAI (default)
LLM_PROVIDER=openai

# Use AWS Bedrock
LLM_PROVIDER=bedrock
```

### Configuring AWS Bedrock

1. **Install AWS dependencies** (already included in requirements.txt):

   ```bash
   pip install langchain-aws boto3
   ```

2. **Set environment variables** in your `.env` file:

   ```bash
   LLM_PROVIDER=bedrock
   AWS_REGION=us-east-1
   BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
   BEDROCK_EMBEDDINGS_MODEL_ID=amazon.titan-embed-text-v2:0
   ```

3. **Configure AWS credentials** using one of these methods:
   - Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - AWS credentials file: `~/.aws/credentials`
   - IAM role (recommended for production/EC2/ECS)

### Supported Bedrock Models

| Model ID | Description |
|----------|-------------|
| `anthropic.claude-3-haiku-20240307-v1:0` | Claude 3 Haiku - Fast and cost-effective |
| `anthropic.claude-3-sonnet-20240229-v1:0` | Claude 3 Sonnet - Balanced performance |
| `amazon.titan-embed-text-v2:0` | Titan Embeddings v2 - Text embeddings |

> **Note:** In production environments, prefer IAM roles over access keys for AWS authentication. Access keys should only be used for local development.

## Project Structure

```
agents-tutorial/
├── config.py                          # Shared configuration module
├── requirements.txt                   # Pinned dependencies
├── .env.example                       # Environment variables template
├── docs/                              # Documentation
├── modulo_01_fundamentals/            # Module 01 - LangChain Fundamentals
├── modulo_02_rag/                     # Module 02 - RAG
├── modulo_03_langgraph/               # Module 03 - LangGraph
├── modulo_04_agentes/                 # Module 04 - Agents
├── modulo_05_multi_agent/             # Module 05 - Multi-Agent
├── modulo_06_producao/                # Module 06 - Production
├── modulo_07_seguranca/               # Module 07 - Security & Reliability
├── modulo_08_avaliacao/               # Module 08 - Evaluation
├── modulo_09_avancados/               # Module 09 - Advanced Concepts
├── modulo_10_mcp/                     # Module 10 - MCP
├── modulo_11_adk/                     # Module 11 - ADK
├── modulo_12_deepeval/                # Module 12 - DeepEval
├── modulo_13_context_engineering/     # Module 13 - Context Engineering
├── projeto_final/                     # Final Project - Corporate Agent
└── tests/                             # Test suite
```

## Modules

### Module 01 — LangChain Fundamentals

Prompt Templates, Structured Output, Tool Calling, Conversation with Memory, Chains & Pipelines, Streaming.

### Module 02 — RAG (Retrieval-Augmented Generation)

Basic RAG, Multi-Document RAG, Metadata Filtering, Query Expansion, Query Decomposition, Hybrid Search, Reranking, Context Compression, Citation-based RAG.

### Module 03 — LangGraph

StateGraph basics, State Management, Conditional Routing, Dynamic Routing, Human-in-the-Loop, Checkpointing, Approval Workflows, Long-Running Workflows.

### Module 04 — Agents

ReAct Agent, Tool Use Agent, Plan & Execute, Reflection Agent, Self-Correction, Task Decomposition, Persistent Memory Agent, Multi-Tool Agent.

### Module 05 — Multi-Agent

Researcher + Writer, Planner + Executor, Supervisor Pattern, Multi-Agent Collaboration, Shared Memory, Agent Delegation, Hierarchical Agents.

### Module 06 — Production

LangSmith Observability, OpenTelemetry, Structured Logging, Correlation IDs, Distributed Tracing, PostgreSQL Persistence, Redis Persistence, Prompt Versioning, Environment Configuration.

### Module 07 — Security & Reliability

Guardrails, Prompt Injection Defense, Output Validation, Rate Limiting, Retry & Backoff, Timeout Management, Circuit Breaker, Fallback Models.

### Module 08 — Evaluation

LLM Evals, RAG Evals, Hallucination Detection, Groundedness Evaluation, Faithfulness Evaluation, Agent Performance Metrics.

### Module 09 — Advanced Concepts

A2A (Agent-to-Agent), Agent Development Lifecycle, Cognitive Loops, Episodic Memory, Semantic Memory, Procedural Memory, Cognitive Architectures, Workflows vs Agents, When NOT to Use Agents.

### Module 10 — MCP (Model Context Protocol)

MCP Server, MCP Client, Custom MCP Tools, MCP + LangChain Agents, MCP Resources, Multi-Server MCP.

### Module 11 — ADK (Google Agent Development Kit)

ADK Fundamentals, Building Agents with ADK, ADK Tools, ADK Sessions, ADK Multi-Agent, ADK vs LangGraph.

### Module 12 — DeepEval (Agent Evaluation Frameworks)

DeepEval Setup & Metrics, Custom Metrics, RAG Evaluation with DeepEval, Agent Evaluation with DeepEval, Benchmarking, CI/CD Integration.

### Module 13 — Context Engineering

Context Window Management, Context Optimization, Dynamic Context Selection, Context Compression, Long-Context Handling, Context-Aware Prompting.

### Final Project — Complete Corporate Agent

A production-ready corporate agent integrating LangGraph, RAG, Persistent Memory, Multi-Agent orchestration, Human-in-the-Loop, Observability, Evaluation, Guardrails, Persistence, and Tool Calling.

## Running Exercises

Each exercise is a standalone Python script. From the module directory:

```bash
python exercicio_XX_name.py
```

For example:

```bash
cd modulo_01_fundamentos
python exercicio_01_prompt_templates.py
```

All exercises import the shared `config.py` from the project root, which handles environment loading and LLM initialization.

## Technology Stack

| Category | Technologies |
|----------|-------------|
| LLM Framework | LangChain, LangGraph, LangFlow, LangSmith |
| LLM Provider | OpenAI (GPT-4o-mini) |
| Embeddings | OpenAI Embeddings (text-embedding-3-small) |
| Vector Stores | ChromaDB |
| MCP | Model Context Protocol SDK |
| ADK | Google Agent Development Kit |
| Evaluation | DeepEval |
| Databases | PostgreSQL (SQLAlchemy), Redis |
| Observability | OpenTelemetry, structlog |
| Validation | Pydantic |
| Tokenization | tiktoken |
| Testing | pytest, hypothesis, pytest-cov, pytest-asyncio |
| Environment | python-dotenv |
| Language | Python 3.11+ |
