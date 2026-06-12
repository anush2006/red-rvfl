"""
RedRVFL model entry point.
"""

# The logic is in src.red_revfl_orchestrator
from src.red_revfl_orchestrator import RedRVFLOrchestrator

def create_model(*args, **kwargs):
    return RedRVFLOrchestrator(*args, **kwargs)
