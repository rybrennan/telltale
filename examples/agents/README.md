# telltale for LLM agents (first test)

The same detector as the federated-learning version, pointed at a population of AI agents. Every agent answers the same fixed probe tasks each round; each answer becomes a probability row; the detector compares each agent with the leave-one-out consensus of its peers and accumulates the divergence with a CUSUM against the agent's own reference rounds. It needs only the agents' answers, not their weights or traces.

This is a first, small test, not an evaluation: 8 runs, one task type, one model family.

## Setup
- **Agents:** 5 support-triage agents on Claude Haiku, each with slightly different instructions (overnight queue, business customers, careful, fast).
- **Probe battery:** 20 fixed support tickets ([`telltale/agents/probes.py`](../../telltale/agents/probes.py)). Each agent labels every ticket with a category, a priority and whether to escalate.
- **Rounds:** 10. Reference rounds 1 to 4. Changes start at round 5.
- **Runs:** 4 healthy (no change); 2 **policy change** (agent 2's instructions quietly edited to file refund and billing questions under account issues and escalate only legal threats); 2 **model update** (agent 2 switched from Haiku to Sonnet).
- **Threshold:** the largest CUSUM any agent reached in the healthy runs after the reference rounds. False alarms are counted by holding each healthy run out in turn.
- Agents run through the Claude Code CLI in headless mode (`claude -p`); every raw reply is cached under `results/agents/`, so scoring is reproducible without calling a model.

## Results ([`results/agent_demo.json`](../../results/agent_demo.json))
| Run | Changed agent flagged? | Rounds after the change | Healthy agents flagged |
|---|---|---|---|
| policy change, run 1 | yes | 2 | none |
| policy change, run 2 | yes | 1 | none |
| model update, run 1 | yes | 2 | none |
| model update, run 2 | **no** (peak 12.1 against a threshold of 13.6) | - | none |

- **False alarms:** holding each healthy run out against the other three, 1 of 4 healthy runs alarmed (5 of its 25 agent-rounds). One healthy run had a much larger natural swing than the others; when it is held out, the threshold from the remaining three is low and it trips. This is the same effect the paper measures: a threshold set from a few healthy runs alarms on roughly one new healthy run in n + 1.
- **All 400 replies parsed.**

## What this does and doesn't show
- It shows the federated-learning detector works on agent answers without modification: three of four planted changes were named within two rounds, with no healthy agent blamed.
- It does not show a detection rate. Eight runs, one task, one model family. A swap to a different model was missed once.
- It can tell that an agent's answers moved, not whether the new answers are wrong.

## Run it
```bash
python scripts/agent_demo.py run     # about 400 agent calls; resumes from cache
python scripts/agent_demo.py score   # thresholds, false alarms, detections
```
