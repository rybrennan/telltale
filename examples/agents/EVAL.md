# Agent drift evaluation

Does a team's normal dashboard notice when one AI agent in a fleet quietly
changes behavior? Does a per-agent monitor?

## Setup

- Two jobs: support-ticket triage (6 agents on Claude Haiku) and expense
  review (6 agents on Claude Sonnet). Each agent has slightly different
  instructions. Every round, every agent labels the same 20 fixed items.
- 10 rounds. From round 5, one agent changes. Which agent rotates across runs.
- Five kinds of change, 6 runs each per job:
  - **policy**: its instructions are edited on a topic the items cover
  - **cautious**: a vague nudge, "when unsure, be more cautious"
  - **memory**: gradual drift, one more biased "note to self" each round
  - **offquiz**: an edit on a topic the items never touch
  - **model**: its model is swapped (Haiku and Sonnet)
- 12 healthy runs per job with no change: 6 set the thresholds, 6 measure
  false alarms. 5,040 agent calls in total, 0 unparsed answers.

Three things are compared on every run:

- **Dashboard**: the fleet-wide escalation rate (triage) or approval rate
  (expenses). It counts as having moved if any round after the change falls
  outside the range seen across all 12 healthy runs.
- **telltale**: per-agent divergence from the other agents, accumulated
  with CUSUM against the agent's own early rounds.
- **Disagreement check**: the share of an agent's answers that differ from
  the other agents' majority, with no accumulation.

## Results

Runs caught out of 6. False alarms on 6 held-out healthy runs per job:
telltale 0 of 360 agent-rounds; disagreement check 1 of 360.

| job | change | dashboard moved | telltale | disagreement | either |
|---|---|---|---|---|---|
| triage | policy | 6 | 5 | 6 | 6 |
| triage | cautious | 2 | 4 | 3 | 4 |
| triage | memory | 1 | 2 | 4 | 4 |
| triage | model | 0 | 2 | 0 | 2 |
| triage | offquiz | 1 | 0 | 0 | 0 |
| expenses | policy | 6 | 6 | 6 | 6 |
| expenses | cautious | 1 | 0 | 0 | 0 |
| expenses | memory | 1 | 1 | 0 | 1 |
| expenses | model | 1 | 2 | 5 | 5 |
| expenses | offquiz | 0 | 0 | 0 | 0 |

## What it shows

- **The dashboard misses subtle drift.** Across the 36 cautious, memory and
  model runs, the fleet-wide rate stayed inside its normal range in 30.
  One changed agent out of six barely moves a fleet average.
- **Per-agent monitoring catches some of what the dashboard misses.** Of
  those 30 quiet runs, telltale or the disagreement check named the changed
  agent in 12, with no false alarms from telltale on healthy runs.
- **Most subtle drift still got past everything.** 18 of those 30 runs went
  undetected. A vague "be more cautious" on the expense agents was invisible
  to all three.
- **telltale is not better than the simple check here.** Caught overall,
  excluding offquiz: telltale 22 of 48, disagreement check 24 of 48. Each wins
  on different changes (telltale on triage model swaps and caution; the
  disagreement check on memory drift and expense model swaps), so running
  both catches more than either.
- **Changes outside the probe items are invisible** (0 of 12), as expected.
  The probes only see behavior they ask about.
- An edit that hits the probe topics directly (policy) is caught by
  everything, the dashboard included.

Small samples: 6 runs per cell. Read the counts as directions, not rates.

## Reproduce

    python scripts/agent_eval.py run --jobs 6   # calls Claude via the claude CLI; cached
    python scripts/agent_eval.py score          # no model calls

Raw replies are in `results/agent_eval/`, scores in `results/agent_eval.json`.
