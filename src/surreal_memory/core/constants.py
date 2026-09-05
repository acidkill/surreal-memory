"""Shared string constants.

``GRAPH_ONLY_PLACEHOLDER`` is the anchor-content tombstone the compression
engine writes when a fiber is reduced to the GRAPH_ONLY tier (see the
``contents_refreshed(..., "[graph-only]", ...)`` call in
``engine/compression.py``). Previously the literal appeared fourteen times
across eight files — a rename in one place silently broke every other check,
which is exactly what this constant exists to make loud (follow-up on the
#193 review).
"""

GRAPH_ONLY_PLACEHOLDER = "[graph-only]"
