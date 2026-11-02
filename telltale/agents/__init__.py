"""Behavioral-regression monitoring for populations of LLM agents.

The federated-learning detector applied to agents: every agent answers the
same fixed probe tasks each round, its answers become a probability array,
and `telltale.detect` flags the agent whose answers drift from its peers and
from its own history.
"""

from telltale.agents.probes import FIELDS, PROBES, answers_to_probs, parse_answers

__all__ = ["FIELDS", "PROBES", "answers_to_probs", "parse_answers"]
