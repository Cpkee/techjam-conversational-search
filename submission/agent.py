"""Submission entry point -- re-exports the real Agent implementation.

The actual agent is implemented across starter/*.py (this bundle's "required local
helper modules", per docs/submission_rules.md): starter/agent.py is the orchestrator;
the other starter/*.py files hold slot extraction, retrieval, fusion, CLARIFY, and the
adaptive-orchestration bandit. See REPORT.md for the method writeup and README.md for
setup/reproduction instructions.
"""
from starter.agent import Agent

__all__ = ["Agent"]
