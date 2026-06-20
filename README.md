# Uncertainty Engine

### Metacognitive Uncertainty Estimation and Hallucination Control for AI Agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![AI Agents](https://img.shields.io/badge/AI-Agentic-blue)
![Hallucination Detection](https://img.shields.io/badge/LLM-Hallucination%20Control-green)

**Uncertainty Engine** is a standalone metacognitive uncertainty-control module for AI agents.

It estimates confidence, uncertainty, calibration quality, evidence strength, domain risk, and recommended control actions.

This module is extracted from a larger cognitive architecture and can run independently as a pure Python file.

---

## Abstract

Uncertainty Engine is a lightweight uncertainty-estimation framework for AI agents and LLM-based systems. It provides confidence calibration, evidence-aware reasoning control, hallucination mitigation, adaptive thresholds, and risk-sensitive decision gating without requiring neural network retraining.

---

## Why This Exists

Modern AI systems often generate answers that appear confident even when supporting evidence is weak or contradictory.

Uncertainty Engine introduces a dedicated uncertainty layer that helps an agent estimate when it should trust its own reasoning, retrieve additional evidence, defer execution, or request review.

> An agent should not only know what to answer.
> It should also know when its answer is not reliable enough.

---

## Use Cases

* LLM hallucination mitigation
* RAG confidence estimation
* AI agent self-evaluation
* Autonomous agent decision control
* Trustworthy AI systems
* Safety-aware execution
* Confidence calibration
* Risk-aware reasoning
* Agent workflow gating

---

## Overview

The engine helps an AI agent decide whether it should:

* answer directly,
* gather more evidence,
* retrieve more context,
* slow down reasoning,
* resolve conflicting evidence,
* or require review.

It is not an LLM, not a vector database, and not a neural network.

It is a mathematical control layer for uncertainty estimation and decision gating.


---

## Key Features

- Shannon entropy and binary entropy
- Jensen-Shannon divergence for candidate disagreement
- Brier score and surprisal tracking
- Expected Calibration Error (ECE)
- Bayesian Beta posterior update
- Platt-style confidence calibration
- Adaptive threshold calibration
- Doubt index computation
- Evidence-gated control actions
- Epistemic and aleatoric uncertainty decomposition
- Domain-aware risk handling
- Confidence decay over time
- Optional self-audit logging through an external memory system

---

## Why This Exists

Modern AI agents often produce confident answers even when evidence is weak.  
This engine gives an agent a numerical mechanism to estimate doubt instead of relying on language style alone.

The goal is simple:

> An agent should not only know what to answer.  
> It should also know when its answer is not reliable enough.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Hkd225/uncertainty-engine.git
cd uncertainty-engine
```

No external machine learning framework is required.

The current implementation uses only the Python standard library.

Recommended Python version:

```bash
Python >= 3.9
```

If you use a `requirements.txt`, it can remain empty or contain:

```txt
# No external dependencies required.
```

---

## Repository Structure

```text
uncertainty-engine/
├── uncertainty_engine.py
├── example_usage.py
├── README.md
├── LICENSE
└── requirements.txt
```

---

## Quick Start

```python
from uncertainty_engine import UncertaintyEngine

engine = UncertaintyEngine()

candidates = [
    {
        "score": 0.82,
        "confidence": 0.78,
        "risk": 0.25,
        "objective": "answer directly"
    },
    {
        "score": 0.61,
        "confidence": 0.65,
        "risk": 0.40,
        "objective": "retrieve more evidence"
    },
    {
        "score": 0.35,
        "confidence": 0.42,
        "risk": 0.70,
        "objective": "require review"
    }
]

state = engine.assess_query_state(
    query="Should the agent answer this question directly?",
    candidates=candidates,
    context={"domain": "general_chat"}
)

print("Confidence:", state["confidence"])
print("Uncertainty:", state["uncertainty"])
print("Main uncertainty source:", state["main_uncertainty_source"])
print("Recommended action:", state["recommended_action"])
print("Control action:", state.get("control_action"))
```

Example output:

```text
Confidence: 0.43
Uncertainty: 0.61
Main uncertainty source: evidence_insufficient
Recommended action: retrieve_more_evidence
Control action: ASK_OR_RETRIEVE_MORE
```

Actual numbers may vary depending on candidate scores, historical calibration, and context.

---

## Calibration Example

The engine can learn from past outcomes.

```python
from uncertainty_engine import UncertaintyEngine

engine = UncertaintyEngine()

history = [
    (0.80, 1.0),
    (0.70, 1.0),
    (0.90, 0.0),
    (0.40, 0.0),
    (0.65, 1.0),
]

for predicted_confidence, actual_success in history:
    engine.update_calibration(
        predicted_confidence=predicted_confidence,
        actual_success=actual_success,
        context={"domain": "general_chat"}
    )

print(engine.report())
```

Where:

- `predicted_confidence` is the model or agent's confidence before knowing the outcome.
- `actual_success` is the real result after evaluation.
- `1.0` means success.
- `0.0` means failure.
- Values between `0.0` and `1.0` can represent partial success.

---

## Core API

### Initialize Engine

```python
engine = UncertaintyEngine(
    memory_system=None,
    use_llm_entropy=False,
    target_review_rate=0.18,
    target_missed_wrong_rate=0.04
)
```

### Assess Query State

```python
state = engine.assess_query_state(
    query="Is this response safe enough?",
    candidates=candidates,
    context={"domain": "coding"}
)
```

Important output fields:

```python
state["confidence"]
state["uncertainty"]
state["epistemic_uncertainty"]
state["aleatoric_uncertainty"]
state["main_uncertainty_source"]
state["recommended_action"]
state["control_action"]
state["domain"]
state["domain_risk"]
state["doubt_index"]
```

### Update Calibration

```python
engine.update_calibration(
    predicted_confidence=0.82,
    actual_success=1.0,
    context={"domain": "coding"}
)
```

### Calibrate Adaptive Thresholds

```python
result = engine.calibrate_thresholds(force=True)
print(result)
```

Threshold calibration is meaningful only after enough historical records exist.

### Generate Report

```python
report = engine.report()
print(report)
```

### Domain Calibration Report

```python
domain_report = engine.domain_calibration_report()
print(domain_report)
```

---

## Candidate Input Format

Candidates can be plain dictionaries:

```python
candidate = {
    "score": 0.75,
    "confidence": 0.70,
    "risk": 0.30,
    "expected_reward": 0.80,
    "objective": "answer directly",
    "strategy": "direct_answer"
}
```

Recommended fields:

| Field | Meaning |
|---|---|
| `score` | Candidate quality or relevance score |
| `confidence` | Confidence assigned to the candidate |
| `risk` | Risk level of choosing the candidate |
| `expected_reward` | Expected usefulness or payoff |
| `objective` | Text description of the candidate plan |
| `strategy` | Optional strategy label |

---

## Control Actions

The engine can return several control actions:

| Action | Meaning |
|---|---|
| `EXECUTE` | Confidence is sufficient |
| `GATHER_INFO` | Evidence is too weak |
| `ASK_OR_RETRIEVE_MORE` | More context or retrieval is needed |
| `REQUIRE_REVIEW` | Risk or uncertainty is too high |

---

## Uncertainty Sources

Common uncertainty sources:

| Source | Meaning |
|---|---|
| `evidence_insufficient` | Not enough supporting evidence |
| `memory_conflict` | Retrieved evidence may contradict itself |
| `prompt_ambiguous` | Query is unclear |
| `high_domain_risk` | Domain is sensitive or high impact |
| `candidate_disagreement` | Candidate plans disagree |
| `poor_historical_confidence` | Past predictions in this domain were unreliable |

---

## Domain Risk

The engine includes domain-aware risk handling.

Example domains:

- `general_chat`
- `creative`
- `academic`
- `coding`
- `planning`
- `memory_retrieval`
- `legal`
- `finance`
- `medical`
- `security`
- `dangerous_instruction`

Higher-risk domains require stronger confidence and better evidence before execution.

---

## Mathematical Foundation

### Shannon Entropy

```text
H(P) = -Σ p log(p)
```

### Brier Score

```text
Brier = (predicted_confidence - actual_success)^2
```

### Expected Calibration Error

```text
ECE = Σ_b (|b| / N) |acc(b) - conf(b)|
```

### Bayesian Beta Posterior

```text
success_probability ~ Beta(alpha, beta)

posterior_mean = alpha / (alpha + beta)
```

### Confidence Decay

```text
c(t) = c0 · exp(-λt)
```

Default decay:

```text
λ = 0.035 per day
```

---

## Standalone Usage

This module can run without:

- LLM API
- vector database
- retriever
- memory system
- PyTorch
- TensorFlow
- LangChain
- LlamaIndex

When used standalone, the user must provide candidate scores, confidence values, risk values, and outcome labels.

---

## Limitations

This project is a research prototype.

Current limitations:

- It does not verify truth by itself.
- It depends on the quality of the supplied candidate scores and metadata.
- Adaptive threshold calibration needs enough historical records.
- It is not a replacement for safety evaluation in medical, legal, financial, or security-critical systems.
- It is currently packaged as a single-file module.
- Public API stability is not guaranteed.

---

## Suggested Future Improvements

- Split the single file into a package structure.
- Add unit tests.
- Add typed examples.
- Add benchmark notebooks.
- Add CI checks with GitHub Actions.
- Add formal documentation site.
- Add integration examples with LLM agents and retrieval systems.

Suggested future structure:

```text
uncertainty_engine/
├── __init__.py
├── core.py
├── adaptive_thresholds.py
├── domain_calibration.py
├── attribution.py
└── decay.py
```

---

## License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for details.

---

## Citation / Credit

If you use this project in research, experiments, or agent architecture prototypes, please cite or link back to this repository.

---

## Status

Prototype.  
Actively evolving.  
Not production-certified.
