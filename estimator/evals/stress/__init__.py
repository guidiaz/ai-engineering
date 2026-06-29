"""Multi-turn stress scenarios for the conversational estimator.

The package drives long conversations over a single project to probe how the
session machinery (sliding window + anchors + cumulative summary + the
metadata extractor) holds onto — or drifts away from — facts stated early.

``scenarios`` defines the conversation profiles; the runner and the
``MemoryDriftMetric`` that scores fact survival are layered on top.
"""
