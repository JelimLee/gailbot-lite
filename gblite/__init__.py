# -*- coding: utf-8 -*-
"""GailBot-Lite: a modern macOS re-implementation of the GailBot
conversation-analysis transcription pipeline."""
from .models import Word, Marker, Utterance
from .analysis import analyze, build_utterances, Thresholds

__version__ = "1.0.0"
