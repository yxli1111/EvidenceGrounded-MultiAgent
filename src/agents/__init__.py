"""Agents package. A owns answer.py, baselines.py, independent_critic.py; D owns revision.py.

Kept import-free on purpose so that importing D's ``src.agents.revision`` does not pull
in A's modules (and vice versa).
"""
