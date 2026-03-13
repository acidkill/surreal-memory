---
title: NeuralMemory Docs
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
license: mit
short_description: Ask questions about NeuralMemory — no LLM, pure neural activation
---

# NeuralMemory Documentation Assistant

Ask questions about [NeuralMemory](https://github.com/nhadaututtheky/neural-memory) — powered by spreading activation, not an LLM.

The brain retrieves relevant documentation using the same neural activation engine that powers `nmem_recall`. No AI hallucinations — only real docs.

## How it works

1. Documentation is pre-encoded into a neural memory brain (neurons + synapses)
2. Your query triggers spreading activation across the knowledge graph
3. The most relevant documentation chunks are retrieved and displayed
4. Confidence score reflects how well the retrieved context matches your query

## Re-training the brain

```bash
# From the repo root
python chatbot/train_docs_brain.py
```

This trains from `docs/`, `README.md`, `CHANGELOG.md`, and `FAQ.md`.

## Local development

```bash
pip install neural-memory gradio
python chatbot/app.py --port 7860
```
