# -*- coding: utf-8 -*-
"""Dataset loading and adaptation module"""

from eval.datasets.hotpotqa import HotpotQABenchmark
from eval.datasets.narrativeqa import NarrativeQABenchmark
from eval.datasets.locomo import LoCoMoBenchmark
from eval.datasets.ruler import RULERBenchmark

__all__ = [
    "HotpotQABenchmark",
    "NarrativeQABenchmark", 
    "LoCoMoBenchmark",
    "RULERBenchmark",
]

