#!/usr/bin/env python3
"""NeuralMemory Documentation Chatbot — Gradio UI.

A self-answering chatbot that uses NeuralMemory's spreading activation
to retrieve relevant documentation. No LLM needed — the brain IS the answer.

Usage (local):
    python chatbot/app.py                     # Launch locally
    python chatbot/app.py --port 7861         # Custom port
    python chatbot/app.py --share             # Create public URL

HuggingFace Spaces: set app_file=app.py, sdk=gradio in README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from neural_memory import BrainConfig
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.storage.sqlite_store import SQLiteStorage

# ── Config ─────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "brain" / "docs.db"

DEPTH_MAP: dict[str, DepthLevel] = {
    "Quick (instant)": DepthLevel.INSTANT,
    "Normal (context)": DepthLevel.CONTEXT,
    "Deep (thorough)": DepthLevel.DEEP,
}

EXAMPLE_QUERIES = [
    "How do I install NeuralMemory?",
    "What is spreading activation?",
    "How do I configure embeddings?",
    "What MCP tools are available?",
    "How does memory consolidation work?",
    "What is a neuron in NeuralMemory?",
    "How to set up cloud sync?",
    "What is the difference between CLI and MCP?",
]

# ── State ──────────────────────────────────────────────────

_storage: SQLiteStorage | None = None
_config: BrainConfig | None = None
_pipeline: ReflexPipeline | None = None


async def get_pipeline() -> ReflexPipeline:
    """Lazy-init the storage + pipeline on first query."""
    global _storage, _config, _pipeline

    if _pipeline is not None:
        return _pipeline

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Brain not found at {DB_PATH}. "
            "Run `python chatbot/train_docs_brain.py` first."
        )

    _storage = SQLiteStorage(str(DB_PATH))
    await _storage.initialize()

    # Load brain config from DB
    brains = await _storage.list_brains()
    if not brains:
        raise ValueError("No brains found in database.")

    brain = brains[0]
    _storage.set_brain(brain.id)
    _config = brain.config or BrainConfig()

    _pipeline = ReflexPipeline(_storage, _config)
    return _pipeline


# ── Query handler ──────────────────────────────────────────


async def answer_query(query: str, depth_label: str) -> tuple[str, str, str]:
    """Query the docs brain and return (answer, confidence_badge, stats).

    Returns:
        Tuple of (context_text, confidence_html, stats_text)
    """
    if not query.strip():
        return "", "", ""

    depth = DEPTH_MAP.get(depth_label, DepthLevel.CONTEXT)

    try:
        pipeline = await get_pipeline()
        result = await pipeline.query(
            query,
            depth=depth,
            max_tokens=1500,
        )
    except FileNotFoundError as e:
        return str(e), "", ""
    except Exception as e:
        return f"Error: {e}", "", ""

    # Format context
    context = result.context or "No relevant documentation found for this query."

    # Confidence badge
    confidence = result.confidence or 0.0
    if confidence >= 0.7:
        badge_color = "#10b981"
        badge_label = "High"
    elif confidence >= 0.4:
        badge_color = "#f59e0b"
        badge_label = "Medium"
    else:
        badge_color = "#ef4444"
        badge_label = "Low"

    badge_html = (
        f'<span style="background:{badge_color};color:white;padding:4px 12px;'
        f'border-radius:12px;font-weight:600;font-size:14px;">'
        f"{badge_label} — {confidence:.0%}</span>"
    )

    # Stats
    latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
    neurons = result.neurons_activated or 0
    stats = f"Depth: {depth.name} | Neurons activated: {neurons} | Latency: {latency}"

    return context, badge_html, stats


# ── Gradio UI ──────────────────────────────────────────────


def create_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="NeuralMemory Docs",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="violet",
        ),
    ) as app:
        gr.Markdown(
            """
# NeuralMemory Documentation Assistant

Ask questions about NeuralMemory — powered by spreading activation, not an LLM.

The brain retrieves relevant documentation using the same neural activation
engine that powers `nmem_recall`. No AI hallucinations — only real docs.
"""
        )

        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Your question",
                    placeholder="How do I install NeuralMemory?",
                    lines=2,
                )
            with gr.Column(scale=1):
                depth_select = gr.Radio(
                    choices=list(DEPTH_MAP.keys()),
                    value="Normal (context)",
                    label="Search depth",
                )

        ask_btn = gr.Button("Ask", variant="primary", size="lg")

        with gr.Row():
            confidence_badge = gr.HTML(label="Confidence")
            stats_text = gr.Textbox(label="Stats", interactive=False)

        answer_output = gr.Markdown(label="Answer")

        gr.Examples(
            examples=[[q] for q in EXAMPLE_QUERIES],
            inputs=[query_input],
            label="Try these questions",
        )

        # Event handlers
        ask_btn.click(
            fn=answer_query,
            inputs=[query_input, depth_select],
            outputs=[answer_output, confidence_badge, stats_text],
        )
        query_input.submit(
            fn=answer_query,
            inputs=[query_input, depth_select],
            outputs=[answer_output, confidence_badge, stats_text],
        )

        gr.Markdown(
            """
---
*Powered by [NeuralMemory](https://github.com/nhadaututtheky/neural-memory)
— brain-inspired persistent memory for AI agents.*
"""
        )

    return app


# ── Main ───────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralMemory Docs Chatbot")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
