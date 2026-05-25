# LangGraph Local Agent

A local, fully offline AI agent built with LangGraph. The agent runs a ReAct-style loop — deciding when to call tools and when to answer directly — powered by a quantized LLM served through Ollama. No external APIs required.

## What it does

The agent accepts natural-language queries in a terminal REPL and can:

| Tool | Description |
|---|---|
| `search_courses` | Keyword-searches a local CSV of university courses and returns matching entries |
| `calc` | Safely evaluates arithmetic expressions |
| `write_text` | Writes text content to a local file (e.g. saves a poem) |
| `search_images` | Searches a local image folder by text description using CLIP embeddings + ChromaDB |

The agent graph is an `agent → tools → agent` loop: the LLM decides whether to call a tool, the `ToolNode` executes it, and the result is fed back for a final answer.

## Tech stack

| Component | Library / Tool |
|---|---|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM runtime | [Ollama](https://ollama.com) (`qwen3:0.6b-q4_K_M`) |
| LLM integration | [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama) |
| Image embeddings | [SentenceTransformers](https://www.sbert.net) — CLIP `clip-ViT-B-32` |
| Vector store | [ChromaDB](https://www.trychroma.com) (persistent, cosine similarity) |
| Image loading | [Pillow](https://python-pillow.org) |
| Language | Python 3.10+ |

## Reproducing

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### 2. Pull the model

```bash
ollama pull qwen3:0.6b-q4_K_M
```

### 3. Install Python dependencies

```bash
pip install langgraph langchain langchain-ollama chromadb sentence-transformers pillow
```

### 4. Add images (optional)

Place `.jpg`, `.jpeg`, `.png`, or `.webp` images into the `images/` folder. On first run the agent will embed them with CLIP and persist the index in `chroma_db/`.

### 5. Run the agent

```bash
python local_agent.py
```

### Example queries

```
You: What machine learning courses are available?
You: What is 128 * 7 + 3?
You: Find me a photo of a dog
You: Write a short poem to poem.txt
```

Type `exit` or `quit` to stop.

## Project structure

```
.
├── local_agent.py   # Agent definition, tools, and run loop
├── courses.csv      # Course dataset used by search_courses
├── images/          # Local images indexed by CLIP
└── chroma_db/       # Auto-generated ChromaDB vector store (gitignored)
```
