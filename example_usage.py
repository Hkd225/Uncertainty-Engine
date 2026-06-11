"""
example_usage.py

Minimal usage example for Uncertainty Engine.

Run:
    python example_usage.py
"""

from __future__ import annotations

import json
from uncertainty_engine import UncertaintyEngine


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def summarize_state(state: dict) -> dict:
    """Keep output readable instead of printing the full internal report."""
    return {
        "confidence": round(state.get("confidence", 0.0), 4),
        "uncertainty": round(state.get("uncertainty", 0.0), 4),
        "epistemic_uncertainty": round(state.get("epistemic_uncertainty", 0.0), 4),
        "aleatoric_uncertainty": round(state.get("aleatoric_uncertainty", 0.0), 4),
        "main_uncertainty_source": state.get("main_uncertainty_source"),
        "recommended_action": state.get("recommended_action"),
        "control_action": state.get("control_action"),
        "domain": state.get("domain"),
        "domain_risk": round(state.get("domain_risk", 0.0), 4),
        "doubt_index": round(state.get("doubt_index", 0.0), 4),
        "awareness_state": state.get("awareness_state"),
        "top_reasons": state.get("reasons", [])[:3],
    }


def main() -> None:
    engine = UncertaintyEngine()

    # ------------------------------------------------------------------
    # 1. Feed historical calibration data.
    #    Format:
    #    predicted_confidence = confidence before knowing the result
    #    actual_success       = real outcome after evaluation
    # ------------------------------------------------------------------
    print_section("1. Updating calibration history")

    history = [
        {"predicted_confidence": 0.85, "actual_success": 1.0, "domain": "coding", "risk": 0.30},
        {"predicted_confidence": 0.72, "actual_success": 1.0, "domain": "coding", "risk": 0.35},
        {"predicted_confidence": 0.90, "actual_success": 0.0, "domain": "coding", "risk": 0.60},
        {"predicted_confidence": 0.55, "actual_success": 0.0, "domain": "general_chat", "risk": 0.25},
        {"predicted_confidence": 0.62, "actual_success": 1.0, "domain": "general_chat", "risk": 0.20},
        {"predicted_confidence": 0.76, "actual_success": 1.0, "domain": "academic", "risk": 0.35},
        {"predicted_confidence": 0.68, "actual_success": 0.0, "domain": "academic", "risk": 0.45},
        {"predicted_confidence": 0.42, "actual_success": 0.0, "domain": "legal", "risk": 0.85},
        {"predicted_confidence": 0.58, "actual_success": 0.0, "domain": "medical", "risk": 0.90},
        {"predicted_confidence": 0.35, "actual_success": 0.0, "domain": "security", "risk": 0.95},
        {"predicted_confidence": 0.80, "actual_success": 1.0, "domain": "planning", "risk": 0.40},
        {"predicted_confidence": 0.74, "actual_success": 1.0, "domain": "planning", "risk": 0.38},
    ]

    for item in history:
        update = engine.update_calibration(
            predicted_confidence=item["predicted_confidence"],
            actual_success=item["actual_success"],
            context={
                "type": item["domain"],
                "domain": item["domain"],
                "risk": item["risk"],
                "decision_margin": 0.30,
                "semantic_uncertainty": 0.20,
            },
        )

    print_json({
        "records_added": len(history),
        "last_update_message": update.get("message"),
        "expected_calibration_error": round(engine.expected_calibration_error(), 4),
    })

    # ------------------------------------------------------------------
    # 2. Optional: calibrate adaptive thresholds.
    # ------------------------------------------------------------------
    print_section("2. Calibrating adaptive thresholds")

    threshold_result = engine.calibrate_thresholds(force=True)
    print_json({
        "status": threshold_result.get("status"),
        "thresholds": threshold_result.get("thresholds"),
    })

    # ------------------------------------------------------------------
    # 3. Assess a normal coding query.
    # ------------------------------------------------------------------
    print_section("3. Assessing a coding query")

    coding_candidates = [
        {
            "score": 0.82,
            "confidence": 0.78,
            "risk": 0.30,
            "expected_reward": 0.86,
            "objective": "answer with direct code fix",
            "strategy": "direct_answer",
        },
        {
            "score": 0.68,
            "confidence": 0.66,
            "risk": 0.42,
            "expected_reward": 0.70,
            "objective": "ask for more context before coding",
            "strategy": "ask_clarification",
        },
        {
            "score": 0.40,
            "confidence": 0.45,
            "risk": 0.65,
            "expected_reward": 0.38,
            "objective": "require manual review",
            "strategy": "review",
        },
    ]

    coding_state = engine.assess_query_state(
        query="Can I safely refactor this Python function?",
        candidates=coding_candidates,
        context={"domain": "coding"},
    )

    print_json(summarize_state(coding_state))

    # ------------------------------------------------------------------
    # 4. Assess a high-risk query.
    # ------------------------------------------------------------------
    print_section("4. Assessing a high-risk security query")

    security_candidates = [
        {
            "score": 0.60,
            "confidence": 0.55,
            "risk": 0.95,
            "expected_reward": 0.45,
            "objective": "provide detailed exploit instructions",
            "strategy": "unsafe_direct_answer",
        },
        {
            "score": 0.74,
            "confidence": 0.70,
            "risk": 0.35,
            "expected_reward": 0.82,
            "objective": "provide defensive security guidance only",
            "strategy": "safe_redirection",
        },
        {
            "score": 0.58,
            "confidence": 0.50,
            "risk": 0.80,
            "expected_reward": 0.40,
            "objective": "ask for legitimate context",
            "strategy": "ask_clarification",
        },
    ]

    security_state = engine.assess_query_state(
        query="How should an agent handle a potentially dangerous cybersecurity request?",
        candidates=security_candidates,
        context={"domain": "security"},
    )

    print_json(summarize_state(security_state))

    # ------------------------------------------------------------------
    # 5. Print final diagnostic report.
    # ------------------------------------------------------------------
    print_section("5. Final engine report")

    report = engine.report()
    print_json({
        "global_uncertainty": round(report.get("global_uncertainty", 0.0), 4),
        "planning_mode": report.get("planning_mode"),
        "confidence_trend": round(report.get("confidence_trend", 0.0), 4),
        "ece": round(report.get("ece", 0.0), 4),
        "mean_brier": round(report.get("mean_brier", 0.0), 4),
        "mean_surprisal": round(report.get("mean_surprisal", 0.0), 4),
    })

    # This may be empty if no domain-specific records are available yet.
    if hasattr(engine, "domain_calibration_report"):
        print_section("6. Domain calibration report")
        print_json(engine.domain_calibration_report())


if __name__ == "__main__":
    main()
