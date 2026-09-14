"""Light test for the Gradio demo module.

Confirms the demo builds a valid Blocks object and its analyze callback handles
the empty-input case. The full pipeline is covered by the orchestrator and API
tests; this just guards that the demo wiring imports and constructs.
"""
import gradio as gr

from app.demo import analyze_image, build_demo


def test_build_demo_returns_blocks():
    demo = build_demo()
    assert isinstance(demo, gr.Blocks)


def test_analyze_image_handles_no_input():
    # Empty input should return gracefully, not raise.
    gallery, message = analyze_image(None)
    assert gallery == []
    assert isinstance(message, str)
