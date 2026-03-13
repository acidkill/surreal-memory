# Documentation Chatbot

NeuralMemory includes a self-answering documentation chatbot powered by spreading activation — no LLM required.

**Try it live:** [HuggingFace Space](https://huggingface.co/spaces/nhadaututtheky/neuralmemory-docs)

## How it works

1. Project documentation is encoded into a neural memory brain (neurons + synapses + fibers)
2. Your query triggers spreading activation across the knowledge graph
3. The most relevant documentation chunks are retrieved and displayed
4. A confidence score reflects how well the context matches your query

The chatbot uses `ReflexPipeline` — the same retrieval engine behind `nmem_recall`.

## Running locally

```bash
pip install neural-memory gradio
python chatbot/app.py
```

Options:

| Flag | Description |
|------|-------------|
| `--port 7861` | Custom port (default: 7860) |
| `--share` | Create a public Gradio URL |

## Re-training the brain

If you've updated the documentation:

```bash
python chatbot/train_docs_brain.py
```

This trains from `docs/`, `README.md`, `CHANGELOG.md`, and `FAQ.md`. The brain is saved to `chatbot/brain/docs.db`.

Training options:

| Flag | Description |
|------|-------------|
| `--brain NAME` | Custom brain name (default: `neuralmemory-docs`) |
| `--export DIR` | Copy the trained DB to another directory |
| `--no-verify` | Skip verification queries |

## Deploying to HuggingFace Spaces

### Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login
```

### One-command deploy

```bash
bash chatbot/deploy_hf.sh
# or with a custom Space name:
bash chatbot/deploy_hf.sh my-org/my-space
```

### Manual deploy

1. Create a new [Gradio Space](https://huggingface.co/new-space) with SDK = Gradio
2. Clone the Space repo
3. Copy `chatbot/app.py`, `chatbot/requirements.txt`, `chatbot/README.md`, and `chatbot/brain/` into the Space
4. Push to HuggingFace

The brain DB is ~51 MB — well within HuggingFace's file size limits.

## Search depth levels

| Level | Pipeline Depth | Speed | Best for |
|-------|---------------|-------|----------|
| Quick | `INSTANT` | ~5ms | Simple keyword lookups |
| Normal | `CONTEXT` | ~20ms | Most questions |
| Deep | `DEEP` | ~50ms | Complex multi-topic queries |
