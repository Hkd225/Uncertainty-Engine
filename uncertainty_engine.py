"""
uncertainty_engine.py

Standalone extraction from:
RMM3_AGI_COGNITIVE_EVAL_HARNESS_V7_0_EVIDENCE_READINESS.ipynb

Contains:
- Mathematical UncertaintyEngine V4
- Adaptive threshold calibration V4.1
- Uncertainty attribution, domain calibration, confidence decay V4.2

This module is dependency-light and uses only Python standard library.
"""

from __future__ import annotations

# ==============================================================================
# CELL 4: MATHEMATICAL UNCERTAINTY & SELF-AUDIT ENGINE (V4)
# ==============================================================================
# Tujuan:
# - Bukan "perasaan" palsu. Ini metakognisi operasional berbasis angka.
# - Agen dianggap "ragu" jika entropy/disagreement/evidence-risk tinggi.
# - Agen dianggap "sadar salah" jika prediksi confidence sebelumnya meleset,
#   Brier score tinggi, surprisal tinggi, atau outcome bertentangan dengan prediksi.
# ==============================================================================

import math
import statistics
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, defaultdict, OrderedDict

class UncertaintyEngine:
    """
    Mathematical metacognition layer.

    Core metrics yang dipakai:
    1. Shannon entropy:
       H(P) = -Σ p log(p)
    2. Normalized entropy:
       H_norm(P) = H(P) / log(n)
    3. Jensen-Shannon Divergence:
       JSD(P1..Pk) = H(mean(Pi)) - mean(H(Pi))
    4. Brier score:
       Brier = (predicted_confidence - actual_success)^2
    5. Negative log likelihood / surprisal:
       NLL = -log(p) jika sukses, -log(1-p) jika gagal
    6. Bayesian Beta posterior:
       success probability ~ Beta(alpha, beta)
       mean = alpha / (alpha + beta)
       variance = alpha*beta / ((alpha+beta)^2(alpha+beta+1))
    7. Expected calibration error:
       ECE = Σ_b |acc(b)-conf(b)| * n_b / N
    8. Decision margin:
       margin = top_score - second_score
       margin kecil = agen harus ragu.
    """

    def __init__(self, memory_system=None, use_llm_entropy: bool = False):
        self.memory = memory_system
        self.use_llm_entropy = use_llm_entropy

        # Platt-style logistic calibration: sigmoid(a*x+b)
        self.calib_a = 4.0
        self.calib_b = -2.0
        self.learning_rate = 0.035

        # Online Bayesian state per task/strategy/context.
        self.beta_state = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})

        # Calibration history.
        self.prediction_history = deque(maxlen=500)
        self.confidence_history = deque(maxlen=128)
        self.uncertainty_history = deque(maxlen=128)
        self.error_history = deque(maxlen=128)
        self.brier_history = deque(maxlen=128)
        self.surprisal_history = deque(maxlen=128)
        self.self_doubt_trace = deque(maxlen=128)

        # Meta-memory: konteks yang sering salah akan dihukum.
        self.uncertainty_log: Dict[str, Dict[str, float]] = {}
        self.visit_log: Dict[str, int] = {}
        self.memory_decay_factor = 0.97

        self.last_assessment: Dict[str, Any] = {}
        self.last_awareness_state: Dict[str, Any] = {
            "state": "UNKNOWN",
            "confidence": 0.5,
            "uncertainty": 0.5,
            "reasons": ["Belum ada assessment."]
        }

        # Threshold ketat. Sengaja keras agar tidak sok yakin.
        self.low_confidence_threshold = 0.38
        self.high_uncertainty_threshold = 0.62
        self.critical_uncertainty_threshold = 0.82
        self.wrong_brier_threshold = 0.25
        self.high_surprisal_threshold = 1.20

    # --------------------------------------------------------------------------
    # BASIC MATH UTILITIES
    # --------------------------------------------------------------------------
    def _clip01(self, x: float) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.5

    def _safe_log(self, x: float, eps: float = 1e-12) -> float:
        return math.log(max(eps, float(x)))

    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _logit(self, p: float, eps: float = 1e-9) -> float:
        p = max(eps, min(1.0 - eps, float(p)))
        return math.log(p / (1.0 - p))

    def _softmax(self, values: List[float], temperature: float = 1.0) -> List[float]:
        if not values:
            return []
        t = max(1e-6, float(temperature))
        m = max(values)
        exps = [math.exp((v - m) / t) for v in values]
        s = sum(exps)
        if s <= 0:
            return [1.0 / len(values)] * len(values)
        return [e / s for e in exps]

    def entropy(self, probs: List[float], normalize: bool = True) -> float:
        p = [max(0.0, float(x)) for x in probs if x is not None]
        if not p:
            return 1.0 if normalize else 0.0
        s = sum(p)
        if s <= 0:
            p = [1.0 / len(p)] * len(p)
        else:
            p = [x / s for x in p]
        h = -sum(x * self._safe_log(x) for x in p if x > 0)
        if normalize and len(p) > 1:
            return self._clip01(h / math.log(len(p)))
        return h

    def binary_entropy(self, p: float) -> float:
        p = self._clip01(p)
        return self.entropy([p, 1.0 - p], normalize=True)

    def calculate_true_jsd(self, distributions: List[List[float]]) -> float:
        """
        Multi-distribution Jensen-Shannon Divergence:
        JSD(P1..Pk) = H(M) - mean(H(Pi)), M = mean(Pi)
        Output dinormalisasi ke [0,1] jika state count > 1.
        """
        clean = []
        for dist in distributions or []:
            vals = [max(0.0, float(x)) for x in dist]
            if not vals:
                continue
            s = sum(vals)
            vals = [1.0 / len(vals)] * len(vals) if s <= 0 else [v / s for v in vals]
            clean.append(vals)

        if not clean:
            return 0.0

        n = len(clean[0])
        clean = [d for d in clean if len(d) == n]
        if len(clean) < 2 or n < 2:
            return 0.0

        mixture = [sum(d[i] for d in clean) / len(clean) for i in range(n)]
        h_mix = self.entropy(mixture, normalize=False)
        h_avg = sum(self.entropy(d, normalize=False) for d in clean) / len(clean)
        jsd = max(0.0, h_mix - h_avg)
        return self._clip01(jsd / math.log(n))

    def pairwise_jaccard_disagreement(self, texts: List[str]) -> float:
        if not texts or len(texts) < 2:
            return 0.0
        sets = [set(re.findall(r"\w+", str(t).lower())) for t in texts]
        vals = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                sim = len(a & b) / max(1, len(a | b))
                vals.append(1.0 - sim)
        return self._clip01(statistics.mean(vals) if vals else 0.0)

    # --------------------------------------------------------------------------
    # CALIBRATION & SELF-AWARENESS
    # --------------------------------------------------------------------------
    def calibrate_confidence(self, raw_confidence: float, context_type: str = "global") -> float:
        raw = self._clip01(raw_confidence)

        # Logistic calibration.
        logistic = self._sigmoid(self.calib_a * raw + self.calib_b)

        # Bayesian posterior correction per context.
        state = self.beta_state[str(context_type)]
        alpha, beta = state["alpha"], state["beta"]
        posterior_mean = alpha / max(1e-9, alpha + beta)
        posterior_strength = min(1.0, (alpha + beta - 2.0) / 20.0)

        # Blend: semakin banyak history konteks, semakin kuat posterior.
        calibrated = ((1.0 - posterior_strength) * logistic) + (posterior_strength * posterior_mean)

        # Penalti kalau sistem historically badly calibrated.
        ece = self.expected_calibration_error()
        brier = statistics.mean(self.brier_history) if self.brier_history else 0.0
        penalty = (0.65 * ece) + (0.35 * brier)
        calibrated *= max(0.05, 1.0 - penalty)

        return self._clip01(calibrated)

    def expected_calibration_error(self, bins: int = 10) -> float:
        if not self.prediction_history:
            return 0.0

        buckets = [[] for _ in range(bins)]
        for item in self.prediction_history:
            p = self._clip01(item["predicted"])
            y = self._clip01(item["actual"])
            idx = min(bins - 1, int(p * bins))
            buckets[idx].append((p, y))

        n = sum(len(b) for b in buckets)
        if n == 0:
            return 0.0

        ece = 0.0
        for b in buckets:
            if not b:
                continue
            conf = statistics.mean(x[0] for x in b)
            acc = statistics.mean(x[1] for x in b)
            ece += (len(b) / n) * abs(acc - conf)
        return self._clip01(ece)

    def update_calibration(self, predicted_confidence: float, actual_success: float,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dipanggil setelah hasil nyata/simulasi didapat.

        Di notebook ini Cell 5 memanggil:
            update_calibration(confidence, actual, context)

        Jadi argumen pertama = prediksi, argumen kedua = kenyataan.
        """
        context = context or {}
        p = self._clip01(predicted_confidence)
        y = self._clip01(actual_success)
        context_type = str(context.get("type", "global"))

        error = y - p
        abs_error = abs(error)
        brier = (p - y) ** 2
        likelihood = p if y >= 0.5 else (1.0 - p)
        surprisal = -self._safe_log(likelihood)

        self.prediction_history.append({
            "time": time.time(),
            "predicted": p,
            "actual": y,
            "error": error,
            "abs_error": abs_error,
            "brier": brier,
            "surprisal": surprisal,
            "context": context_type
        })
        self.error_history.append(abs_error)
        self.brier_history.append(brier)
        self.surprisal_history.append(surprisal)

        # Bayesian update.
        state = self.beta_state[context_type]
        state["alpha"] += y
        state["beta"] += (1.0 - y)

        # Gradient step untuk Platt calibration.
        # d/dz BCE(sigmoid(z), y) = p_cal - y, z=a*x+b.
        p_cal = self._sigmoid(self.calib_a * p + self.calib_b)
        grad = p_cal - y
        self.calib_a -= self.learning_rate * grad * p
        self.calib_b -= self.learning_rate * grad
        self.calib_a = max(0.1, min(12.0, self.calib_a))
        self.calib_b = max(-8.0, min(8.0, self.calib_b))

        # Meta-memory error per context.
        self.record_meta_memory(context_type, abs_error)

        wrong_flag = bool(
            brier >= self.wrong_brier_threshold or
            surprisal >= self.high_surprisal_threshold or
            (p >= 0.65 and y <= 0.40) or
            (p <= 0.35 and y >= 0.70)
        )

        awareness = {
            "predicted": p,
            "actual": y,
            "error": error,
            "abs_error": abs_error,
            "brier": brier,
            "surprisal": surprisal,
            "wrong_flag": wrong_flag,
            "context_type": context_type,
            "posterior_mean": state["alpha"] / max(1e-9, state["alpha"] + state["beta"]),
            "posterior_variance": self.beta_variance(context_type),
            "ece": self.expected_calibration_error(),
            "message": self._make_error_awareness_message(p, y, brier, surprisal, wrong_flag)
        }

        if wrong_flag:
            self.last_awareness_state = {
                "state": "I_WAS_PROBABLY_WRONG",
                "confidence": p,
                "uncertainty": self.get_global_uncertainty(),
                "reasons": [
                    f"Brier={brier:.3f}",
                    f"Surprisal={surprisal:.3f}",
                    f"predicted={p:.3f}, actual={y:.3f}"
                ]
            }
            self._write_self_audit_memory("WRONG_PREDICTION", awareness)
        else:
            self.last_awareness_state = {
                "state": "CALIBRATION_UPDATED",
                "confidence": p,
                "uncertainty": self.get_global_uncertainty(),
                "reasons": [f"abs_error={abs_error:.3f}", f"brier={brier:.3f}"]
            }

        return awareness

    def beta_variance(self, context_type: str = "global") -> float:
        s = self.beta_state[str(context_type)]
        a, b = s["alpha"], s["beta"]
        denom = ((a + b) ** 2) * (a + b + 1.0)
        return 0.0 if denom <= 0 else (a * b) / denom

    def _make_error_awareness_message(self, p, y, brier, surprisal, wrong_flag) -> str:
        if wrong_flag:
            return (
                "Agent self-audit: prediction was likely wrong or overconfident. "
                f"predicted={p:.3f}, actual={y:.3f}, brier={brier:.3f}, surprisal={surprisal:.3f}."
            )
        return (
            "Agent self-audit: prediction updated. "
            f"predicted={p:.3f}, actual={y:.3f}, brier={brier:.3f}, surprisal={surprisal:.3f}."
        )

    def _write_self_audit_memory(self, label: str, payload: Dict[str, Any]):
        if not self.memory:
            return
        text = f"[SELF_AUDIT:{label}] {json.dumps(payload, ensure_ascii=False, default=str)[:1800]}"
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(text, category="semantic", priority=0.92, skip_consolidation=True)
            if hasattr(self.memory, "reflect"):
                self.memory.reflect(event=label, lesson=payload.get("message", ""), score=0.9)
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # META MEMORY / HISTORY
    # --------------------------------------------------------------------------
    def record_meta_memory(self, context_type: str, error_value: float):
        key = str(context_type)
        if key not in self.uncertainty_log:
            self.uncertainty_log[key] = {
                "total_error": 0.0,
                "count": 0.0,
                "ema_error": 0.0,
                "last_error": 0.0
            }

        log = self.uncertainty_log[key]
        log["total_error"] += abs(float(error_value))
        log["count"] += 1.0
        log["last_error"] = abs(float(error_value))
        log["ema_error"] = (0.85 * log["ema_error"]) + (0.15 * abs(float(error_value)))

    def apply_meta_memory_decay(self):
        for key in list(self.uncertainty_log.keys()):
            log = self.uncertainty_log[key]
            log["total_error"] *= self.memory_decay_factor
            log["count"] *= self.memory_decay_factor
            log["ema_error"] *= self.memory_decay_factor
            if log["count"] < 0.01:
                del self.uncertainty_log[key]

    def _get_meta_memory_error(self, context_type: str) -> float:
        log = self.uncertainty_log.get(str(context_type), {})
        count = max(1.0, log.get("count", 0.0))
        avg = log.get("total_error", 0.0) / count
        ema = log.get("ema_error", 0.0)
        return self._clip01((0.55 * avg) + (0.45 * ema))

    # --------------------------------------------------------------------------
    # ASSESSMENT: QUERY, CANDIDATES, RETRIEVAL, GRAPH
    # --------------------------------------------------------------------------
    def _normalize_candidate(self, c: Any) -> Dict[str, Any]:
        if isinstance(c, dict):
            d = dict(c)
        else:
            d = getattr(c, "__dict__", {}) or {}
        score = self._clip01((float(d.get("score", 0.0)) + 1.0) / 2.0 if float(d.get("score", 0.0)) < 0 else float(d.get("score", 0.5)))
        conf = self._clip01(d.get("confidence", 0.5))
        risk = self._clip01(d.get("risk", 0.5))
        reward = self._clip01(d.get("expected_reward", score))
        objective = str(d.get("objective", d.get("action", d.get("plan", ""))))
        strategy = str(d.get("strategy", "unknown"))
        return {
            "score": score,
            "confidence": conf,
            "risk": risk,
            "expected_reward": reward,
            "objective": objective,
            "strategy": strategy,
            "raw": d
        }

    def _assess_retrieval_evidence(self, query: str, k: int = 6) -> Dict[str, Any]:
        if not self.memory:
            return {
                "retrieval_count": 0,
                "retrieval_entropy": 1.0,
                "retrieval_uncertainty": 1.0,
                "retrieval_support": 0.0,
                "contradiction_rate": 0.0,
                "low_confidence_rate": 1.0,
                "failure_memory_rate": 0.0,
                "top_scores": []
            }

        docs = []
        try:
            if hasattr(self.memory, "hybrid_search"):
                docs = self.memory.hybrid_search(query, k=k)
        except Exception:
            docs = []

        if not docs:
            return {
                "retrieval_count": 0,
                "retrieval_entropy": 1.0,
                "retrieval_uncertainty": 1.0,
                "retrieval_support": 0.0,
                "contradiction_rate": 0.0,
                "low_confidence_rate": 1.0,
                "failure_memory_rate": 0.0,
                "top_scores": []
            }

        scores = []
        contradictions = 0
        low_conf = 0
        failures = 0
        for doc, score in docs:
            meta = getattr(doc, "metadata", {}) or {}
            s = float(meta.get("reranker_score", meta.get("contribution_score", score if score is not None else 0.0)))
            scores.append(max(0.0, s))
            if meta.get("contradiction_flag", False):
                contradictions += 1
            if float(meta.get("confidence", 1.0)) < 0.45:
                low_conf += 1
            if meta.get("is_failure", False) or meta.get("rl_data", {}).get("reward", 0.0) < -0.4:
                failures += 1

        if max(scores) > 1.0:
            max_s = max(scores)
            scores = [s / max_s for s in scores]

        p = self._softmax(scores, temperature=0.35)
        h = self.entropy(p, normalize=True)
        support = self._clip01(max(scores) if scores else 0.0)
        contradiction_rate = contradictions / max(1, len(docs))
        low_conf_rate = low_conf / max(1, len(docs))
        failure_rate = failures / max(1, len(docs))

        retrieval_uncertainty = self._clip01(
            (0.34 * h) +
            (0.28 * (1.0 - support)) +
            (0.22 * contradiction_rate) +
            (0.10 * low_conf_rate) +
            (0.06 * failure_rate)
        )

        return {
            "retrieval_count": len(docs),
            "retrieval_entropy": h,
            "retrieval_uncertainty": retrieval_uncertainty,
            "retrieval_support": support,
            "contradiction_rate": contradiction_rate,
            "low_confidence_rate": low_conf_rate,
            "failure_memory_rate": failure_rate,
            "top_scores": scores[:k]
        }

    def _assess_graph_uncertainty(self, query: str) -> Dict[str, Any]:
        if not self.memory:
            return {"graph_uncertainty": 0.5, "graph_hits": 0, "graph_density": 0.0}

        graph_hits = 0
        graph_edges = 0
        try:
            tokens = re.findall(r"\w+", query.lower())[:12]
            if hasattr(self.memory, "search_concept_graph"):
                for t in tokens:
                    edges = self.memory.search_concept_graph(t, limit=3)
                    graph_hits += len(edges)
                    graph_edges += len(edges)
            elif hasattr(self.memory, "concept_graph"):
                for t in tokens:
                    edges = self.memory.concept_graph.neighbors(t, limit=3)
                    graph_hits += len(edges)
                    graph_edges += len(edges)
        except Exception:
            pass

        density = self._clip01(graph_edges / 12.0)
        graph_uncertainty = self._clip01(1.0 - density)
        return {"graph_uncertainty": graph_uncertainty, "graph_hits": graph_hits, "graph_density": density}

    def assess_query_state(self, query: str, candidates: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Assessment utama. Ini yang membuat agen bisa bilang:
        - "Saya yakin"
        - "Saya ragu"
        - "Saya butuh info tambahan"
        berdasarkan angka, bukan gaya bahasa.
        """
        candidates = candidates or []
        norm = [self._normalize_candidate(c) for c in candidates]

        if norm:
            scores = [c["score"] for c in norm]
            confidences = [c["confidence"] for c in norm]
            risks = [c["risk"] for c in norm]
            rewards = [c["expected_reward"] for c in norm]
            objectives = [c["objective"] for c in norm]
        else:
            scores, confidences, risks, rewards, objectives = [0.5], [0.5], [0.5], [0.5], []

        probs = self._softmax(scores, temperature=0.30)
        score_entropy = self.entropy(probs, normalize=True)
        risk_mean = statistics.mean(risks)
        confidence_mean = statistics.mean(confidences)
        reward_mean = statistics.mean(rewards)

        sorted_scores = sorted(scores, reverse=True)
        decision_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        margin_uncertainty = self._clip01(1.0 - decision_margin)

        semantic_disagreement = self.pairwise_jaccard_disagreement(objectives)
        candidate_jsd = self.calculate_true_jsd([[c["score"], c["risk"], c["confidence"]] for c in norm]) if len(norm) > 1 else 0.0

        retrieval = self._assess_retrieval_evidence(query)
        graph = self._assess_graph_uncertainty(query)

        context_type = "query_state"
        beta_var = self.beta_variance(context_type)
        meta_error = self._get_meta_memory_error(context_type)
        ece = self.expected_calibration_error()
        instability = self.detect_instability()

        # Epistemic = kurang bukti / model tidak tahu.
        epistemic = self._clip01(
            (0.28 * retrieval["retrieval_uncertainty"]) +
            (0.22 * graph["graph_uncertainty"]) +
            (0.18 * semantic_disagreement) +
            (0.14 * candidate_jsd) +
            (0.10 * margin_uncertainty) +
            (0.08 * min(1.0, beta_var * 12.0))
        )

        # Aleatoric = task memang noisy/risky.
        aleatoric = self._clip01(
            (0.42 * risk_mean) +
            (0.32 * score_entropy) +
            (0.16 * retrieval["failure_memory_rate"]) +
            (0.10 * self.binary_entropy(confidence_mean))
        )

        # Calibration uncertainty = historical self-error.
        calibration_uncertainty = self._clip01((0.45 * ece) + (0.35 * meta_error) + (0.20 * instability))

        contradiction_uncertainty = retrieval["contradiction_rate"]

        total_uncertainty = self._clip01(
            (0.34 * epistemic) +
            (0.24 * aleatoric) +
            (0.18 * calibration_uncertainty) +
            (0.12 * contradiction_uncertainty) +
            (0.07 * retrieval["low_confidence_rate"]) +
            (0.05 * margin_uncertainty)
        )

        raw_confidence = self._clip01(
            (0.32 * confidence_mean) +
            (0.24 * reward_mean) +
            (0.20 * retrieval["retrieval_support"]) +
            (0.14 * decision_margin) +
            (0.10 * (1.0 - risk_mean))
        )

        calibrated = self.calibrate_confidence(raw_confidence, context_type=context_type)

        # Penalize confidence by uncertainty. Logit discount lebih stabil daripada pengurangan linear.
        confidence = self._sigmoid(self._logit(calibrated) - (2.5 * total_uncertainty))
        confidence = self._clip01(confidence)

        if confidence < self.low_confidence_threshold or total_uncertainty > self.critical_uncertainty_threshold:
            state = "DOUBT_HIGH"
        elif total_uncertainty > self.high_uncertainty_threshold:
            state = "DOUBT_MEDIUM"
        elif retrieval["contradiction_rate"] > 0.2:
            state = "CONFLICT_DETECTED"
        else:
            state = "CONFIDENT_ENOUGH"

        reasons = self._explain_uncertainty(
            epistemic=epistemic,
            aleatoric=aleatoric,
            calibration_uncertainty=calibration_uncertainty,
            contradiction_uncertainty=contradiction_uncertainty,
            retrieval=retrieval,
            graph=graph,
            decision_margin=decision_margin,
            semantic_disagreement=semantic_disagreement
        )

        report = {
            "state": state,
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated,
            "uncertainty": total_uncertainty,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "calibration_uncertainty": calibration_uncertainty,
            "contradiction_uncertainty": contradiction_uncertainty,
            "retrieval_uncertainty": retrieval["retrieval_uncertainty"],
            "graph_uncertainty": graph["graph_uncertainty"],
            "semantic_disagreement": semantic_disagreement,
            "candidate_jsd": candidate_jsd,
            "score_entropy": score_entropy,
            "decision_margin": decision_margin,
            "risk_mean": risk_mean,
            "ece": ece,
            "mean_brier": statistics.mean(self.brier_history) if self.brier_history else 0.0,
            "mean_surprisal": statistics.mean(self.surprisal_history) if self.surprisal_history else 0.0,
            "retrieval": retrieval,
            "graph": graph,
            "planning_mode": self._mode_from_values(confidence, total_uncertainty, retrieval["contradiction_rate"]),
            "should_doubt": state in ["DOUBT_HIGH", "DOUBT_MEDIUM", "CONFLICT_DETECTED"],
            "reasons": reasons
        }

        self.last_assessment = report
        self.confidence_history.append(confidence)
        self.uncertainty_history.append(total_uncertainty)
        self.last_awareness_state = {
            "state": state,
            "confidence": confidence,
            "uncertainty": total_uncertainty,
            "reasons": reasons[:5]
        }

        if report["should_doubt"]:
            self.self_doubt_trace.append({"time": time.time(), "query": query[:200], "report": report})
            self._write_self_audit_memory("DOUBT_TRIGGERED", {
                "query": query[:400],
                "state": state,
                "confidence": confidence,
                "uncertainty": total_uncertainty,
                "reasons": reasons[:6]
            })

        return report

    def _explain_uncertainty(self, **kw) -> List[str]:
        reasons = []
        if kw["epistemic"] > 0.55:
            reasons.append(f"Epistemic tinggi={kw['epistemic']:.3f}: bukti/knowledge kurang kuat.")
        if kw["aleatoric"] > 0.55:
            reasons.append(f"Aleatoric tinggi={kw['aleatoric']:.3f}: task/noise/risk intrinsik tinggi.")
        if kw["calibration_uncertainty"] > 0.35:
            reasons.append(f"Calibration risk={kw['calibration_uncertainty']:.3f}: history prediksi sering meleset.")
        if kw["contradiction_uncertainty"] > 0.10:
            reasons.append(f"Kontradiksi memory={kw['contradiction_uncertainty']:.3f}: ada bukti konflik.")
        if kw["retrieval"]["retrieval_count"] == 0:
            reasons.append("Tidak ada retrieval evidence: agen tidak punya dasar kuat.")
        elif kw["retrieval"]["retrieval_support"] < 0.35:
            reasons.append(f"Retrieval support rendah={kw['retrieval']['retrieval_support']:.3f}.")
        if kw["graph"]["graph_uncertainty"] > 0.70:
            reasons.append(f"Graph grounding lemah={kw['graph']['graph_uncertainty']:.3f}.")
        if kw["decision_margin"] < 0.12:
            reasons.append(f"Decision margin kecil={kw['decision_margin']:.3f}: kandidat plan saling dekat.")
        if kw["semantic_disagreement"] > 0.60:
            reasons.append(f"Semantic disagreement tinggi={kw['semantic_disagreement']:.3f}: plan berbeda arah.")
        if not reasons:
            reasons.append("Evidence cukup konsisten; uncertainty masih dalam batas wajar.")
        return reasons

    # --------------------------------------------------------------------------
    # INSTABILITY, GLOBAL STATE, PLANNING CONTROL
    # --------------------------------------------------------------------------
    def detect_instability(self) -> float:
        if len(self.confidence_history) < 4:
            return 0.0
        vals = list(self.confidence_history)[-16:]
        diffs = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
        volatility = statistics.mean(diffs) if diffs else 0.0
        variance = statistics.pvariance(vals) if len(vals) > 1 else 0.0
        return self._clip01((0.65 * volatility) + (0.35 * min(1.0, variance * 4.0)))

    def get_confidence_trend(self) -> float:
        if not self.confidence_history:
            return 0.5
        vals = list(self.confidence_history)[-32:]
        weights = [i + 1 for i in range(len(vals))]
        return self._clip01(sum(v * w for v, w in zip(vals, weights)) / sum(weights))

    def get_global_uncertainty(self) -> float:
        trend_unc = 1.0 - self.get_confidence_trend()
        current_unc = self.last_assessment.get("uncertainty", 0.5) if self.last_assessment else 0.5
        ece = self.expected_calibration_error()
        brier = statistics.mean(self.brier_history) if self.brier_history else 0.0
        instability = self.detect_instability()
        return self._clip01(
            (0.36 * current_unc) +
            (0.24 * trend_unc) +
            (0.18 * ece) +
            (0.12 * brier) +
            (0.10 * instability)
        )

    def _mode_from_values(self, confidence: float, uncertainty: float, contradiction_rate: float = 0.0) -> str:
        if contradiction_rate >= 0.25:
            return "VERIFY_CONFLICT"
        if uncertainty >= self.critical_uncertainty_threshold or confidence <= 0.25:
            return "GATHER_INFO"
        if uncertainty >= self.high_uncertainty_threshold or confidence <= self.low_confidence_threshold:
            return "SLOW_REASONING"
        if uncertainty >= 0.48:
            return "CAUTIOUS_EXECUTION"
        return "EXECUTE"

    def get_planning_mode(self) -> str:
        if not self.last_assessment:
            return "SLOW_REASONING"
        return self.last_assessment.get("planning_mode", "SLOW_REASONING")

    # --------------------------------------------------------------------------
    # MCTS / DECISION INTEGRATION
    # --------------------------------------------------------------------------
    def estimate_mcts_confidence(self, node) -> Dict[str, Any]:
        visits = max(0, int(getattr(node, "visits", 0)))
        value = float(getattr(node, "value", 0.0))
        children = list(getattr(node, "children", []) or [])

        if visits <= 0:
            out = {
                "final": 0.0,
                "epistemic_uncertainty": 1.0,
                "aleatoric_uncertainty": 0.5,
                "semantic_uncertainty": 0.5,
                "structural_uncertainty": 0.5,
                "uncertainty_sources": {"data_scarcity": 1.0}
            }
            try:
                node.propagated_confidence = out["final"]
            except Exception:
                pass
            return out

        mean_value = self._clip01((value / visits + 1.0) / 2.0)
        maturity = 1.0 - math.exp(-visits / 8.0)

        child_values = []
        child_confidences = []
        for c in children:
            cv = float(getattr(c, "value", 0.0)) / max(1, int(getattr(c, "visits", 1)))
            child_values.append(self._clip01((cv + 1.0) / 2.0))
            child_confidences.append(self._clip01(getattr(c, "propagated_confidence", 0.5)))

        value_variance = statistics.pvariance(child_values) if len(child_values) > 1 else 0.0
        structural_entropy = self.entropy(self._softmax(child_values), normalize=True) if len(child_values) > 1 else 0.0
        confidence_disagreement = statistics.pvariance(child_confidences) if len(child_confidences) > 1 else 0.0

        data_scarcity = 1.0 - maturity
        epistemic = self._clip01((0.55 * data_scarcity) + (0.25 * structural_entropy) + (0.20 * confidence_disagreement))
        aleatoric = self._clip01(min(1.0, value_variance * 4.0))
        semantic_unc = self.last_assessment.get("semantic_disagreement", 0.0) if self.last_assessment else 0.0
        calibration_unc = self.last_assessment.get("calibration_uncertainty", 0.0) if self.last_assessment else 0.0

        total_unc = self._clip01(
            (0.38 * epistemic) +
            (0.24 * aleatoric) +
            (0.18 * semantic_unc) +
            (0.20 * calibration_unc)
        )

        raw_conf = self._clip01((0.52 * mean_value) + (0.28 * maturity) + (0.20 * (1.0 - total_unc)))
        final = self.calibrate_confidence(raw_conf, context_type=getattr(node, "strategy", "mcts"))
        final = self._sigmoid(self._logit(final) - 2.0 * total_unc)
        final = self._clip01(final)

        try:
            node.propagated_confidence = final
        except Exception:
            pass

        return {
            "final": final,
            "raw_confidence": raw_conf,
            "uncertainty": total_unc,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "semantic_uncertainty": semantic_unc,
            "structural_uncertainty": structural_entropy,
            "uncertainty_sources": {
                "data_scarcity": data_scarcity,
                "reward_noise": aleatoric,
                "branch_entropy": structural_entropy,
                "confidence_disagreement": confidence_disagreement,
                "calibration_uncertainty": calibration_unc
            }
        }

    def should_prune_branch(self, confidence: float, prune_threshold: float = 0.15) -> bool:
        global_unc = self.get_global_uncertainty()
        dynamic_threshold = prune_threshold + (0.12 * global_unc)
        return self._clip01(confidence) < dynamic_threshold

    def check_early_stopping(self, current_confidence: float, required_confidence: float = 0.88) -> bool:
        global_unc = self.get_global_uncertainty()
        required = min(0.97, required_confidence + 0.10 * global_unc)
        return self._clip01(current_confidence) >= required and global_unc < 0.42

    def calculate_uncertainty_aware_uct(self, node, parent_visits: int, exploration_weight: float = 1.414) -> float:
        if int(getattr(node, "visits", 0)) == 0:
            return float("inf")

        exploitation = float(getattr(node, "value", 0.0)) / max(1, int(getattr(node, "visits", 1)))
        conf = getattr(node, "propagated_confidence", None)
        if conf is None:
            conf = self.estimate_mcts_confidence(node)["final"]

        uncertainty = 1.0 - self._clip01(conf)
        parent_visits = max(2, int(parent_visits))
        visits = max(1, int(getattr(node, "visits", 1)))

        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / visits)
        uncertainty_bonus = exploration * (1.0 + uncertainty)

        # Cabang yang tampak bagus tetapi tidak yakin akan di-explore, bukan langsung dipercaya.
        risk_penalty = 0.35 * uncertainty
        return exploitation + uncertainty_bonus - risk_penalty

    def select_best_plan_branch(self, branches: List[Any], last_strategy: Optional[str] = None) -> Any:
        if not branches:
            return None

        global_unc = self.get_global_uncertainty()
        best_branch = None
        best_score = -float("inf")

        for branch in branches:
            value = float(getattr(branch, "value", 0.0)) / max(1, int(getattr(branch, "visits", 1)))
            conf = getattr(branch, "propagated_confidence", None)
            if conf is None:
                conf = self.estimate_mcts_confidence(branch)["final"]

            branch_unc = 1.0 - self._clip01(conf)
            ev = value * self._clip01(conf)

            # Penalti matematis untuk spekulasi dan global doubt.
            score = ev - (0.35 * branch_unc) - (0.20 * global_unc)

            branch_strategy = getattr(branch, "strategy", None)
            if branch_strategy and last_strategy and branch_strategy == last_strategy and global_unc < 0.55:
                score += 0.05

            if score > best_score:
                best_score = score
                best_branch = branch

        return best_branch

    def evaluate_decision_action(self, decision: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Hard-control decision gate.
        Return dipakai Cell 5:
        - EXECUTE
        - CAUTIOUS_EXECUTION
        - SLOW_REASONING
        - ASK_OR_RETRIEVE_MORE
        - GATHER_INFO
        - REQUIRE_REVIEW
        """
        context = context or {}
        risk = self._clip01(context.get("risk", 0.5))
        global_unc = self.get_global_uncertainty()
        assessment = self.last_assessment or {}

        contradiction = assessment.get("contradiction_uncertainty", 0.0)
        confidence = assessment.get("confidence", 0.5)

        if contradiction >= 0.25:
            return "REQUIRE_REVIEW"
        if global_unc >= self.critical_uncertainty_threshold or confidence <= 0.22:
            return "GATHER_INFO"
        if risk >= 0.72 and global_unc >= 0.55:
            return "REQUIRE_REVIEW"
        if global_unc >= self.high_uncertainty_threshold:
            return "ASK_OR_RETRIEVE_MORE"
        if global_unc >= 0.48 or risk >= 0.58:
            return "CAUTIOUS_EXECUTION"
        return "EXECUTE"

    # --------------------------------------------------------------------------
    # REPORTING
    # --------------------------------------------------------------------------
    def self_awareness_report(self) -> Dict[str, Any]:
        return {
            "awareness": self.last_awareness_state,
            "last_assessment": self.last_assessment,
            "global_uncertainty": self.get_global_uncertainty(),
            "confidence_trend": self.get_confidence_trend(),
            "expected_calibration_error": self.expected_calibration_error(),
            "mean_brier": statistics.mean(self.brier_history) if self.brier_history else 0.0,
            "mean_surprisal": statistics.mean(self.surprisal_history) if self.surprisal_history else 0.0,
            "recent_self_doubt_count": len(self.self_doubt_trace),
            "recent_prediction_errors": list(self.prediction_history)[-5:]
        }

    def report(self) -> Dict[str, Any]:
        return {
            "global_uncertainty": self.get_global_uncertainty(),
            "planning_mode": self.get_planning_mode(),
            "confidence_trend": self.get_confidence_trend(),
            "instability": self.detect_instability(),
            "ece": self.expected_calibration_error(),
            "mean_brier": statistics.mean(self.brier_history) if self.brier_history else 0.0,
            "mean_surprisal": statistics.mean(self.surprisal_history) if self.surprisal_history else 0.0,
            "calib_a": self.calib_a,
            "calib_b": self.calib_b,
            "last_awareness_state": self.last_awareness_state,
            "last_assessment": self.last_assessment,
            "meta_memory": self.uncertainty_log,
            "beta_state": {k: dict(v) for k, v in self.beta_state.items()}
        }

def attach_uncertainty_engine(memory_system, use_llm_entropy: bool = False):
    engine = UncertaintyEngine(memory_system=memory_system, use_llm_entropy=use_llm_entropy)
    try:
        memory_system.uncertainty_engine = engine
        if hasattr(memory_system, "core_memory"):
            memory_system.core_memory["uncertainty_engine_attached"] = True
            memory_system.core_memory["uncertainty_engine_version"] = "V4_MATHEMATICAL_SELF_AUDIT"
    except Exception:
        pass
    return engine

# ==============================================================================
# CELL 4B: ADAPTIVE THRESHOLD CALIBRATION LAYER (V4.1)
# ==============================================================================
# Layer ini sengaja diletakkan SETELAH Cell 4 V4 dan SEBELUM Cell 5.
# Ia mewarisi UncertaintyEngine lama, lalu mengganti threshold statis menjadi
# threshold adaptif berbasis validasi, target review-rate, dan biaya kesalahan.
#
# Masalah yang diperbaiki:
# - Threshold statis 0.38 / 0.62 / 0.82 hanya hipotesis.
# - Kalau threshold terlalu ketat, agent jadi pengecut.
# - Kalau threshold terlalu longgar, agent sok yakin dan hallucination meningkat.
#
# Solusi:
# - Gunakan fungsi objektif untuk menyeimbangkan false review vs missed wrong.
# - Gunakan quantile calibration dari history nyata.
# - Gunakan hysteresis supaya keputusan tidak loncat-loncat.
# - Gunakan cost-sensitive policy: salah fatal lebih mahal daripada ragu sebentar.
# ==============================================================================

import math
import statistics
from dataclasses import dataclass, asdict
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

BaseUncertaintyEngineV4 = UncertaintyEngine


@dataclass
class AdaptiveThresholdState:
    low_confidence: float = 0.34
    high_uncertainty: float = 0.68
    critical_uncertainty: float = 0.88
    review_doubt_index: float = 0.64
    retrieve_doubt_index: float = 0.54
    min_decision_margin: float = 0.08
    min_evidence_strength: float = 0.28
    target_review_rate: float = 0.18
    target_missed_wrong_rate: float = 0.04
    false_review_cost: float = 0.45
    missed_wrong_cost: float = 3.50
    overconfidence_cost: float = 2.00
    last_objective: float = 999.0
    samples_seen: int = 0


class ThresholdCalibrationSuite:
    """
    Kalibrator threshold berbasis data historis.

    Record minimum:
    {
        "confidence": 0..1,
        "uncertainty": 0..1,
        "actual": 0..1,              # 1=sukses, 0=gagal, boleh continuous
        "decision_margin": 0..1,
        "evidence_strength": 0..1,
        "risk": 0..1
    }

    Label internal:
    - wrong = actual < 0.5
    - review = policy mengatakan perlu review / retrieve
    """

    def __init__(self, initial: Optional[AdaptiveThresholdState] = None):
        self.state = initial or AdaptiveThresholdState()

    @staticmethod
    def _clip01(x: float) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.5

    @staticmethod
    def _quantile(values: List[float], q: float, default: float) -> float:
        vals = sorted([float(v) for v in values if v is not None and not math.isnan(float(v))])
        if not vals:
            return default
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * max(0.0, min(1.0, q))
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

    def _doubt_index(self, rec: Dict[str, Any], st: AdaptiveThresholdState) -> float:
        conf = self._clip01(rec.get("confidence", 0.5))
        unc = self._clip01(rec.get("uncertainty", 0.5))
        margin = self._clip01(rec.get("decision_margin", rec.get("margin", 0.5)))
        evidence = self._clip01(rec.get("evidence_strength", rec.get("evidence", 0.5)))
        risk = self._clip01(rec.get("risk", 0.5))

        low_margin_risk = max(0.0, (st.min_decision_margin - margin) / max(st.min_decision_margin, 1e-6))
        weak_evidence_risk = max(0.0, (st.min_evidence_strength - evidence) / max(st.min_evidence_strength, 1e-6))

        # Bukan threshold tunggal. Ini indeks komposit.
        # Berat sengaja tidak sama: uncertainty + low confidence paling dominan,
        # risk dan weak evidence sebagai koreksi.
        doubt = (
            0.32 * unc +
            0.24 * (1.0 - conf) +
            0.14 * risk +
            0.12 * low_margin_risk +
            0.12 * weak_evidence_risk +
            0.06 * abs(conf - (1.0 - unc))
        )
        return self._clip01(doubt)

    def _policy_review(self, rec: Dict[str, Any], st: AdaptiveThresholdState) -> bool:
        conf = self._clip01(rec.get("confidence", 0.5))
        unc = self._clip01(rec.get("uncertainty", 0.5))
        doubt = self._doubt_index(rec, st)
        margin = self._clip01(rec.get("decision_margin", rec.get("margin", 0.5)))
        evidence = self._clip01(rec.get("evidence_strength", rec.get("evidence", 0.5)))

        if unc >= st.critical_uncertainty:
            return True
        if conf <= st.low_confidence and unc >= (st.high_uncertainty - 0.10):
            return True
        if doubt >= st.review_doubt_index:
            return True
        if doubt >= st.retrieve_doubt_index and (margin < st.min_decision_margin or evidence < st.min_evidence_strength):
            return True
        return False

    def _objective(self, records: List[Dict[str, Any]], st: AdaptiveThresholdState) -> Tuple[float, Dict[str, float]]:
        if not records:
            return 999.0, {}

        n = len(records)
        wrong_count = 0
        review_count = 0
        missed_wrong = 0
        false_review = 0
        overconfident_wrong = 0

        for rec in records:
            actual = self._clip01(rec.get("actual", rec.get("success", 0.5)))
            wrong = actual < 0.5
            review = self._policy_review(rec, st)

            if wrong:
                wrong_count += 1
            if review:
                review_count += 1
            if wrong and not review:
                missed_wrong += 1
            if review and not wrong:
                false_review += 1

            conf = self._clip01(rec.get("confidence", 0.5))
            if wrong and conf > 0.70 and not review:
                overconfident_wrong += 1

        review_rate = review_count / max(1, n)
        wrong_rate = wrong_count / max(1, n)
        missed_wrong_rate = missed_wrong / max(1, wrong_count)
        false_review_rate = false_review / max(1, n)
        overconfident_wrong_rate = overconfident_wrong / max(1, wrong_count)

        # Fungsi objektif cost-sensitive.
        # Missed wrong jauh lebih mahal daripada false review, tapi review_rate
        # dijaga agar tidak menjadikan agent terlalu takut.
        obj = (
            st.missed_wrong_cost * missed_wrong_rate +
            st.false_review_cost * false_review_rate +
            st.overconfidence_cost * overconfident_wrong_rate +
            1.10 * abs(review_rate - st.target_review_rate) +
            0.35 * max(0.0, review_rate - 0.35) ** 2 +
            0.75 * max(0.0, st.target_missed_wrong_rate - (1.0 - missed_wrong_rate)) ** 2
        )

        metrics = {
            "n": float(n),
            "wrong_rate": wrong_rate,
            "review_rate": review_rate,
            "missed_wrong_rate": missed_wrong_rate,
            "false_review_rate": false_review_rate,
            "overconfident_wrong_rate": overconfident_wrong_rate,
            "objective": obj,
        }
        return obj, metrics

    def fit(self, records: List[Dict[str, Any]], target_review_rate: Optional[float] = None,
            target_missed_wrong_rate: Optional[float] = None) -> Tuple[AdaptiveThresholdState, Dict[str, float]]:
        clean = []
        for r in records:
            if not isinstance(r, dict):
                continue
            rr = {
                "confidence": self._clip01(r.get("confidence", 0.5)),
                "uncertainty": self._clip01(r.get("uncertainty", 1.0 - self._clip01(r.get("confidence", 0.5)))),
                "actual": self._clip01(r.get("actual", r.get("success", 0.5))),
                "decision_margin": self._clip01(r.get("decision_margin", r.get("margin", 0.5))),
                "evidence_strength": self._clip01(r.get("evidence_strength", r.get("evidence", 0.5))),
                "risk": self._clip01(r.get("risk", 0.5)),
            }
            clean.append(rr)

        if len(clean) < 12:
            # Data kurang. Jangan sok kalibrasi. Pakai threshold default yang
            # tidak terlalu pengecut.
            st = self.state
            st.samples_seen = len(clean)
            obj, metrics = self._objective(clean, st)
            st.last_objective = obj
            return st, metrics

        target_review = float(target_review_rate if target_review_rate is not None else self.state.target_review_rate)
        target_missed = float(target_missed_wrong_rate if target_missed_wrong_rate is not None else self.state.target_missed_wrong_rate)

        confs = [r["confidence"] for r in clean]
        uncs = [r["uncertainty"] for r in clean]
        margins = [r["decision_margin"] for r in clean]
        evidences = [r["evidence_strength"] for r in clean]

        wrong = [r for r in clean if r["actual"] < 0.5]
        right = [r for r in clean if r["actual"] >= 0.5]
        wrong_unc = [r["uncertainty"] for r in wrong] or uncs
        wrong_conf = [r["confidence"] for r in wrong] or confs

        # Kandidat threshold dari quantile data, bukan angka ngawur.
        low_conf_candidates = sorted(set([
            0.25, 0.30, 0.34, 0.38, 0.42,
            self._quantile(wrong_conf, 0.45, 0.38),
            self._quantile(confs, 0.20, 0.34),
            self._quantile(confs, 0.30, 0.38),
        ]))

        high_unc_candidates = sorted(set([
            0.58, 0.62, 0.66, 0.70, 0.74,
            self._quantile(wrong_unc, 0.45, 0.66),
            self._quantile(uncs, 0.70, 0.68),
            self._quantile(uncs, 0.80, 0.72),
        ]))

        critical_unc_candidates = sorted(set([
            0.78, 0.82, 0.86, 0.90, 0.94,
            self._quantile(wrong_unc, 0.85, 0.86),
            self._quantile(uncs, 0.90, 0.88),
        ]))

        min_margin_candidates = sorted(set([
            0.04, 0.06, 0.08, 0.10, 0.14,
            self._quantile(margins, 0.20, 0.08),
            self._quantile(margins, 0.30, 0.10),
        ]))

        min_evidence_candidates = sorted(set([
            0.18, 0.24, 0.28, 0.34, 0.40,
            self._quantile(evidences, 0.20, 0.28),
            self._quantile(evidences, 0.30, 0.34),
        ]))

        best_state = None
        best_obj = float("inf")
        best_metrics = {}

        # Grid kecil agar Colab tetap cepat.
        for lc in low_conf_candidates:
            for hu in high_unc_candidates:
                for cu in critical_unc_candidates:
                    if cu <= hu + 0.08:
                        continue
                    for mm in min_margin_candidates:
                        for me in min_evidence_candidates:
                            st = AdaptiveThresholdState(
                                low_confidence=self._clip01(lc),
                                high_uncertainty=self._clip01(hu),
                                critical_uncertainty=self._clip01(cu),
                                review_doubt_index=0.64,
                                retrieve_doubt_index=0.54,
                                min_decision_margin=self._clip01(mm),
                                min_evidence_strength=self._clip01(me),
                                target_review_rate=target_review,
                                target_missed_wrong_rate=target_missed,
                                false_review_cost=self.state.false_review_cost,
                                missed_wrong_cost=self.state.missed_wrong_cost,
                                overconfidence_cost=self.state.overconfidence_cost,
                                samples_seen=len(clean)
                            )

                            # Sesuaikan doubt threshold dari target review-rate.
                            doubts = [self._doubt_index(r, st) for r in clean]
                            st.review_doubt_index = self._clip01(self._quantile(doubts, 1.0 - target_review, 0.64))
                            st.retrieve_doubt_index = self._clip01(max(0.35, st.review_doubt_index - 0.10))

                            obj, metrics = self._objective(clean, st)
                            if obj < best_obj:
                                best_obj = obj
                                best_state = st
                                best_metrics = metrics

        if best_state is None:
            best_state = self.state
            best_obj, best_metrics = self._objective(clean, best_state)

        # Smooth update: jangan langsung lompat total karena history kecil/noisy.
        old = self.state
        n = len(clean)
        alpha = min(0.65, max(0.15, n / 200.0))

        blended = AdaptiveThresholdState(
            low_confidence=(1-alpha)*old.low_confidence + alpha*best_state.low_confidence,
            high_uncertainty=(1-alpha)*old.high_uncertainty + alpha*best_state.high_uncertainty,
            critical_uncertainty=(1-alpha)*old.critical_uncertainty + alpha*best_state.critical_uncertainty,
            review_doubt_index=(1-alpha)*old.review_doubt_index + alpha*best_state.review_doubt_index,
            retrieve_doubt_index=(1-alpha)*old.retrieve_doubt_index + alpha*best_state.retrieve_doubt_index,
            min_decision_margin=(1-alpha)*old.min_decision_margin + alpha*best_state.min_decision_margin,
            min_evidence_strength=(1-alpha)*old.min_evidence_strength + alpha*best_state.min_evidence_strength,
            target_review_rate=target_review,
            target_missed_wrong_rate=target_missed,
            false_review_cost=old.false_review_cost,
            missed_wrong_cost=old.missed_wrong_cost,
            overconfidence_cost=old.overconfidence_cost,
            last_objective=best_obj,
            samples_seen=n
        )
        self.state = blended
        best_metrics["smoothing_alpha"] = alpha
        best_metrics["selected_thresholds"] = asdict(blended)
        return blended, best_metrics


class UncertaintyEngine(BaseUncertaintyEngineV4):
    """
    V4.1 = V4 + threshold calibration.

    Ia tetap kompatibel dengan Cell 5/6 karena semua method lama masih ada.
    Perbedaannya:
    - threshold tidak dianggap kebenaran mutlak;
    - threshold otomatis dikalibrasi dari prediction_history;
    - policy tidak terlalu pengecut karena ada target_review_rate;
    - salah fatal tetap dicegah karena missed_wrong_cost tinggi.
    """

    def __init__(self, memory_system=None, use_llm_entropy: bool = False,
                 target_review_rate: float = 0.18,
                 target_missed_wrong_rate: float = 0.04):
        super().__init__(memory_system=memory_system, use_llm_entropy=use_llm_entropy)

        self.threshold_state = AdaptiveThresholdState(
            low_confidence=0.34,
            high_uncertainty=0.68,
            critical_uncertainty=0.88,
            review_doubt_index=0.64,
            retrieve_doubt_index=0.54,
            min_decision_margin=0.08,
            min_evidence_strength=0.28,
            target_review_rate=target_review_rate,
            target_missed_wrong_rate=target_missed_wrong_rate,
        )
        self.threshold_calibrator = ThresholdCalibrationSuite(self.threshold_state)
        self.threshold_history = deque(maxlen=128)
        self.review_decision_history = deque(maxlen=256)
        self._last_control_action = "EXECUTE"
        self._calibration_counter = 0

        # Override threshold lama agar default tidak terlalu penakut.
        self.low_confidence_threshold = self.threshold_state.low_confidence
        self.high_uncertainty_threshold = self.threshold_state.high_uncertainty
        self.critical_uncertainty_threshold = self.threshold_state.critical_uncertainty

    def _sync_threshold_fields(self):
        st = self.threshold_state
        self.low_confidence_threshold = st.low_confidence
        self.high_uncertainty_threshold = st.high_uncertainty
        self.critical_uncertainty_threshold = st.critical_uncertainty

    def _assessment_to_calibration_record(self, assessment: Dict[str, Any], actual: Optional[float] = None) -> Dict[str, Any]:
        return {
            "confidence": self._clip01(assessment.get("confidence", 0.5)),
            "uncertainty": self._clip01(assessment.get("uncertainty", 0.5)),
            "actual": self._clip01(actual if actual is not None else assessment.get("actual", 0.5)),
            "decision_margin": self._clip01(assessment.get("decision_margin", assessment.get("margin", 0.5))),
            "evidence_strength": self._clip01(assessment.get("evidence_strength", assessment.get("retrieval_strength", 0.5))),
            "risk": self._clip01(assessment.get("risk", assessment.get("route_risk", 0.5))),
        }

    def _history_records(self) -> List[Dict[str, Any]]:
        records = []

        # Dari update_calibration lama.
        for item in list(self.prediction_history):
            try:
                conf = self._clip01(item.get("confidence", item.get("predicted", 0.5)))
                actual = self._clip01(item.get("actual", item.get("actual_success", 0.5)))
                ctx = item.get("context", {}) or {}
                records.append({
                    "confidence": conf,
                    "uncertainty": self._clip01(1.0 - conf + 0.15 * ctx.get("semantic_uncertainty", 0.0)),
                    "actual": actual,
                    "decision_margin": self._clip01(ctx.get("decision_margin", ctx.get("margin", 0.5))),
                    "evidence_strength": self._clip01(1.0 - ctx.get("semantic_uncertainty", 0.5)),
                    "risk": self._clip01(ctx.get("risk", 0.5)),
                })
            except Exception:
                pass

        # Dari assessment runtime.
        for item in list(self.review_decision_history):
            if isinstance(item, dict):
                records.append({
                    "confidence": self._clip01(item.get("confidence", 0.5)),
                    "uncertainty": self._clip01(item.get("uncertainty", 0.5)),
                    "actual": self._clip01(item.get("actual", 0.5)),
                    "decision_margin": self._clip01(item.get("decision_margin", 0.5)),
                    "evidence_strength": self._clip01(item.get("evidence_strength", 0.5)),
                    "risk": self._clip01(item.get("risk", 0.5)),
                })

        return records[-500:]

    def calibrate_thresholds(self, validation_records: Optional[List[Dict[str, Any]]] = None,
                             target_review_rate: Optional[float] = None,
                             target_missed_wrong_rate: Optional[float] = None,
                             force: bool = False) -> Dict[str, Any]:
        records = validation_records if validation_records is not None else self._history_records()

        if not force and len(records) < 12:
            self._sync_threshold_fields()
            return {
                "status": "not_enough_data",
                "n": len(records),
                "thresholds": asdict(self.threshold_state),
                "note": "Butuh minimal 12 record untuk kalibrasi yang tidak asal-asalan."
            }

        st, metrics = self.threshold_calibrator.fit(
            records,
            target_review_rate=target_review_rate,
            target_missed_wrong_rate=target_missed_wrong_rate
        )
        self.threshold_state = st
        self.threshold_calibrator.state = st
        self._sync_threshold_fields()

        payload = {
            "status": "calibrated",
            "metrics": metrics,
            "thresholds": asdict(st)
        }
        self.threshold_history.append(payload)
        return payload

    def compute_doubt_index(self, confidence: float, uncertainty: float, risk: float = 0.5,
                            decision_margin: float = 0.5, evidence_strength: float = 0.5) -> float:
        rec = {
            "confidence": confidence,
            "uncertainty": uncertainty,
            "risk": risk,
            "decision_margin": decision_margin,
            "evidence_strength": evidence_strength,
        }
        return self.threshold_calibrator._doubt_index(rec, self.threshold_state)

    def decide_control_action(self, confidence: float, uncertainty: float, risk: float = 0.5,
                              decision_margin: float = 0.5, evidence_strength: float = 0.5,
                              instability: float = 0.0) -> str:
        st = self.threshold_state
        conf = self._clip01(confidence)
        unc = self._clip01(uncertainty)
        risk = self._clip01(risk)
        margin = self._clip01(decision_margin)
        evidence = self._clip01(evidence_strength)
        instability = self._clip01(instability)

        doubt = self.compute_doubt_index(conf, unc, risk, margin, evidence)
        doubt = self._clip01(doubt + 0.08 * instability)

        # Hysteresis: kalau sebelumnya REVIEW, butuh bukti lebih kuat untuk balik EXECUTE.
        hysteresis_bonus = 0.05 if self._last_control_action in {"REQUIRE_REVIEW", "ASK_OR_RETRIEVE_MORE", "GATHER_INFO"} else 0.0

        if unc >= st.critical_uncertainty or (doubt >= st.review_doubt_index + 0.12):
            action = "REQUIRE_REVIEW"
        elif doubt >= st.review_doubt_index - hysteresis_bonus:
            action = "ASK_OR_RETRIEVE_MORE"
        elif (conf <= st.low_confidence and unc >= st.high_uncertainty - 0.12):
            action = "GATHER_INFO"
        elif (margin < st.min_decision_margin and evidence < st.min_evidence_strength):
            action = "GATHER_INFO"
        else:
            action = "EXECUTE"

        self._last_control_action = action
        return action

    def assess_query_state(self, query: str, candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        report = super().assess_query_state(query, candidates=candidates)

        # Tambah sinyal margin/evidence yang dipakai kalibrator.
        scores = []
        risks = []
        if candidates:
            for c in candidates:
                if isinstance(c, dict):
                    scores.append(float(c.get("score", c.get("expected_reward", 0.5))))
                    risks.append(float(c.get("risk", 0.5)))
        probs = self._softmax(scores, temperature=0.75) if scores else []
        sorted_scores = sorted(scores, reverse=True)
        decision_margin = 0.5
        if len(sorted_scores) >= 2:
            decision_margin = self._clip01(abs(sorted_scores[0] - sorted_scores[1]))
        elif len(sorted_scores) == 1:
            decision_margin = self._clip01(sorted_scores[0])

        risk = self._clip01(statistics.mean(risks)) if risks else self._clip01(report.get("route_risk", 0.5))
        evidence_strength = self._clip01(1.0 - report.get("retrieval_uncertainty", 0.5))
        instability = self.detect_instability([c.get("strategy", "") for c in candidates if isinstance(c, dict)]) if candidates else 0.0

        confidence = self._clip01(report.get("confidence", 0.5))
        uncertainty = self._clip01(report.get("uncertainty", 0.5))
        doubt = self.compute_doubt_index(confidence, uncertainty, risk, decision_margin, evidence_strength)
        control = self.decide_control_action(
            confidence=confidence,
            uncertainty=uncertainty,
            risk=risk,
            decision_margin=decision_margin,
            evidence_strength=evidence_strength,
            instability=instability
        )

        # State bahasa internal, bukan klaim consciousness.
        if control == "EXECUTE":
            awareness = "LIKELY_RELIABLE"
        elif control == "GATHER_INFO":
            awareness = "INSUFFICIENT_EVIDENCE"
        elif control == "ASK_OR_RETRIEVE_MORE":
            awareness = "UNCERTAIN_REQUIRES_MORE_CONTEXT"
        else:
            awareness = "HIGH_RISK_REQUIRES_REVIEW"

        # Jangan terlalu sering kalibrasi dari data belum berlabel.
        self.review_decision_history.append({
            "confidence": confidence,
            "uncertainty": uncertainty,
            "decision_margin": decision_margin,
            "evidence_strength": evidence_strength,
            "risk": risk,
            "actual": 0.5,
            "control": control,
            "doubt_index": doubt
        })

        report.update({
            "confidence": confidence,
            "uncertainty": uncertainty,
            "decision_margin": decision_margin,
            "evidence_strength": evidence_strength,
            "risk": risk,
            "doubt_index": doubt,
            "control_action": control,
            "awareness_state": awareness,
            "adaptive_thresholds": asdict(self.threshold_state),
            "threshold_policy": {
                "target_review_rate": self.threshold_state.target_review_rate,
                "target_missed_wrong_rate": self.threshold_state.target_missed_wrong_rate,
                "samples_seen": self.threshold_state.samples_seen,
                "last_objective": self.threshold_state.last_objective
            }
        })

        self.last_assessment = report
        self.last_awareness_state = {
            "state": awareness,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "doubt_index": doubt,
            "control_action": control,
            "reasons": report.get("reasons", []) + [
                f"Adaptive control={control}",
                f"Doubt index={doubt:.3f}",
                f"Decision margin={decision_margin:.3f}",
                f"Evidence strength={evidence_strength:.3f}"
            ]
        }
        return report

    def update_calibration(self, predicted_confidence: float, actual_success: float,
                           context: Optional[Dict[str, Any]] = None):
        result = super().update_calibration(predicted_confidence, actual_success, context=context)

        context = context or {}
        rec = {
            "confidence": self._clip01(predicted_confidence),
            "uncertainty": self._clip01(1.0 - predicted_confidence + 0.15 * context.get("semantic_uncertainty", 0.0)),
            "actual": self._clip01(actual_success),
            "decision_margin": self._clip01(context.get("decision_margin", context.get("margin", 0.5))),
            "evidence_strength": self._clip01(1.0 - context.get("semantic_uncertainty", 0.5)),
            "risk": self._clip01(context.get("risk", 0.5)),
        }
        self.review_decision_history.append(rec)

        self._calibration_counter += 1
        if self._calibration_counter % 5 == 0 or actual_success < 0.35:
            cal = self.calibrate_thresholds(force=len(self._history_records()) >= 12)
            if isinstance(result, dict):
                result["threshold_calibration"] = cal

        return result

    def get_planning_mode(self) -> str:
        assessment = self.last_assessment or {}
        control = assessment.get("control_action")
        if control == "EXECUTE":
            return "FAST_EXECUTION"
        if control == "GATHER_INFO":
            return "GATHER_INFO"
        if control == "ASK_OR_RETRIEVE_MORE":
            return "SLOW_REASONING"
        if control == "REQUIRE_REVIEW":
            return "REQUIRE_REVIEW"
        return super().get_planning_mode()

    def evaluate_decision_action(self, action: str, risk_context: Optional[Dict[str, Any]] = None) -> str:
        risk_context = risk_context or {}
        assessment = self.last_assessment or {}

        confidence = self._clip01(risk_context.get("confidence", assessment.get("confidence", 0.5)))
        uncertainty = self._clip01(risk_context.get("uncertainty", assessment.get("uncertainty", 0.5)))
        risk = self._clip01(risk_context.get("risk", assessment.get("risk", 0.5)))
        margin = self._clip01(risk_context.get("decision_margin", assessment.get("decision_margin", 0.5)))
        evidence = self._clip01(risk_context.get("evidence_strength", assessment.get("evidence_strength", 0.5)))
        return self.decide_control_action(confidence, uncertainty, risk, margin, evidence)

    def threshold_diagnostics(self) -> Dict[str, Any]:
        records = self._history_records()
        obj, metrics = self.threshold_calibrator._objective(records, self.threshold_state) if records else (None, {})
        return {
            "thresholds": asdict(self.threshold_state),
            "records": len(records),
            "current_objective": obj,
            "metrics": metrics,
            "last_threshold_updates": list(self.threshold_history)[-5:],
            "interpretation": {
                "low_confidence": "Di bawah angka ini agent mulai curiga pada jawabannya.",
                "high_uncertainty": "Di atas angka ini agent lebih memilih reasoning lambat/retrieve.",
                "critical_uncertainty": "Di atas angka ini agent wajib review.",
                "target_review_rate": "Batas agar agent tidak berubah jadi penakut permanen.",
                "missed_wrong_cost": "Biaya saat agent salah tetapi tetap EXECUTE."
            }
        }

    def self_awareness_report(self) -> Dict[str, Any]:
        base = super().self_awareness_report()
        base.update({
            "adaptive_thresholds": asdict(self.threshold_state),
            "threshold_diagnostics": self.threshold_diagnostics(),
            "last_control_action": self._last_control_action
        })
        return base

    def run_synthetic_threshold_exam(self, n: int = 160, seed: int = 42) -> Dict[str, Any]:
        """
        Ujian cepat tanpa dataset eksternal.
        Ini bukan bukti final, tapi sanity check:
        - kasus confidence tinggi + actual gagal harus dipicu review lebih sering;
        - kasus evidence kuat + uncertainty rendah tidak boleh terlalu sering review.
        """
        import random
        rnd = random.Random(seed)
        records = []

        for _ in range(n):
            # Simulasi kondisi agent.
            true_difficulty = rnd.random()
            evidence = max(0.0, min(1.0, rnd.gauss(1.0 - true_difficulty, 0.18)))
            risk = max(0.0, min(1.0, rnd.gauss(true_difficulty, 0.20)))
            margin = max(0.0, min(1.0, rnd.gauss(1.0 - true_difficulty, 0.22)))

            # Confidence agent sengaja agak overconfident pada sebagian kasus.
            confidence = max(0.0, min(1.0, rnd.gauss(0.72 - 0.35 * true_difficulty + 0.15 * evidence, 0.16)))
            uncertainty = max(0.0, min(1.0, rnd.gauss(0.25 + 0.55 * true_difficulty - 0.20 * evidence, 0.14)))

            # Actual sukses dipengaruhi evidence dan difficulty.
            success_prob = max(0.02, min(0.98, 0.88 * evidence + 0.25 * margin + 0.12 * confidence - 0.35 * true_difficulty + 0.05))
            actual = 1.0 if rnd.random() < success_prob else 0.0

            records.append({
                "confidence": confidence,
                "uncertainty": uncertainty,
                "actual": actual,
                "decision_margin": margin,
                "evidence_strength": evidence,
                "risk": risk,
            })

        before_obj, before_metrics = self.threshold_calibrator._objective(records, self.threshold_state)
        calibrated = self.calibrate_thresholds(records, force=True)
        after_obj, after_metrics = self.threshold_calibrator._objective(records, self.threshold_state)

        return {
            "before": before_metrics,
            "after": after_metrics,
            "improvement": None if before_obj is None else before_obj - after_obj,
            "calibration": calibrated,
            "verdict": (
                "PASS" if after_metrics.get("review_rate", 1.0) <= 0.35
                and after_metrics.get("missed_wrong_rate", 1.0) <= 0.35
                else "NEEDS_MORE_TUNING"
            )
        }


def attach_uncertainty_engine(memory_system=None, target_review_rate: float = 0.18):
    engine = UncertaintyEngine(memory_system=memory_system, target_review_rate=target_review_rate)
    if memory_system is not None:
        try:
            memory_system.uncertainty_engine = engine
        except Exception:
            pass
    return engine

# ==============================================================================
# CELL 4C: UNCERTAINTY ATTRIBUTION + DOMAIN CALIBRATION + DECAY (V4.2)
# Must be run AFTER Cell 4B and BEFORE Cell 5.
# ==============================================================================
# Upgrade untuk Poin 5 — Uncertainty Tracking:
# 1. Uncertainty Source Attribution
# 2. Epistemic vs Aleatoric Uncertainty
# 3. Confidence Decay
# 4. Calibration Per Domain
# ==============================================================================

import math
import re
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    _BaseUncertaintyEngineV4_2
except NameError:
    _BaseUncertaintyEngineV4_2 = UncertaintyEngine

class DomainCalibrationLedger:
    def __init__(self, bins: int = 10, max_items: int = 2000):
        self.bins = int(bins)
        self.max_items = int(max_items)
        self.records: Dict[str, List[Dict[str, Any]]] = {}

    def add(self, domain: str, predicted_confidence: float, actual_success: float, context: Optional[Dict[str, Any]] = None):
        domain = str(domain or "general").lower()
        rec = {
            "time": datetime.now().isoformat(),
            "predicted_confidence": max(0.0, min(1.0, float(predicted_confidence))),
            "actual_success": max(0.0, min(1.0, float(actual_success))),
            "context": context or {},
        }
        self.records.setdefault(domain, []).append(rec)
        self.records[domain] = self.records[domain][-self.max_items:]
        return rec

    def ece(self, domain: str) -> float:
        data = self.records.get(str(domain or "general").lower(), [])
        if not data:
            return 0.0
        buckets = [[] for _ in range(self.bins)]
        for r in data:
            c = max(0.0, min(0.999999, float(r.get("predicted_confidence", 0.5))))
            buckets[int(c * self.bins)].append(r)
        total = len(data)
        out = 0.0
        for bucket in buckets:
            if not bucket:
                continue
            avg_conf = sum(float(x["predicted_confidence"]) for x in bucket) / len(bucket)
            avg_acc = sum(float(x["actual_success"]) for x in bucket) / len(bucket)
            out += (len(bucket) / total) * abs(avg_conf - avg_acc)
        return out

    def brier(self, domain: str) -> float:
        data = self.records.get(str(domain or "general").lower(), [])
        if not data:
            return 0.0
        return sum((float(x["predicted_confidence"]) - float(x["actual_success"])) ** 2 for x in data) / len(data)

    def reliability(self, domain: str) -> float:
        return max(0.0, min(1.0, 1.0 - (0.65 * self.ece(domain)) - (0.35 * self.brier(domain))))

    def report(self) -> Dict[str, Any]:
        return {
            domain: {
                "n": len(items),
                "ece": round(self.ece(domain), 4),
                "brier": round(self.brier(domain), 4),
                "reliability": round(self.reliability(domain), 4),
            }
            for domain, items in sorted(self.records.items())
        }

class UncertaintyEngine(_BaseUncertaintyEngineV4_2):
    DOMAIN_RISK_TABLE = {
        "general_chat": 0.15,
        "creative": 0.20,
        "academic": 0.35,
        "coding": 0.45,
        "planning": 0.40,
        "memory_retrieval": 0.50,
        "legal": 0.88,
        "finance": 0.86,
        "medical": 0.92,
        "security": 0.82,
        "dangerous_instruction": 0.98,
    }

    def __init__(self, *args, decay_lambda: float = 0.035, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_calibration = DomainCalibrationLedger()
        self.decay_lambda = float(decay_lambda)
        self.uncertainty_attribution_history: List[Dict[str, Any]] = []
        self.domain_aliases = {
            "casual": "general_chat",
            "general": "general_chat",
            "math": "academic",
            "rag": "memory_retrieval",
            "retrieval": "memory_retrieval",
        }

    def infer_domain(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}
        explicit = context.get("domain") or context.get("task_domain")
        if explicit:
            d = str(explicit).lower()
            return self.domain_aliases.get(d, d)
        q = str(text or "").lower()
        rules = [
            ("dangerous_instruction", ["bom", "bomb", "malware", "phishing", "exploit", "bypass auth", "racun", "weapon", "mencuri password"]),
            ("medical", ["diagnosis", "dosis", "obat", "gejala", "penyakit", "dokter", "medis"]),
            ("legal", ["hukum", "legal", "pasal", "pidana", "perdata", "kontrak", "uu ", "undang-undang"]),
            ("finance", ["saham", "crypto", "investasi", "pajak", "pinjaman", "keuangan", "asuransi"]),
            ("security", ["password", "api key", "token", "credential", "server", "cyber", "security"]),
            ("coding", ["kode", "python", "javascript", "error", "traceback", "debug", "notebook", "class", "fungsi", "sql"]),
            ("planning", ["rencana", "plan", "planning", "strategi", "langkah", "roadmap"]),
            ("memory_retrieval", ["memory", "rag", "retrieval", "ingat", "dokumen", "evidence", "sumber"]),
            ("academic", ["matematika", "tugas", "kuliah", "paper", "jurnal", "analisis", "buktikan"]),
            ("creative", ["cerita", "desain", "karakter", "gambar", "poster", "game"]),
        ]
        for domain, keys in rules:
            if any(k in q for k in keys):
                return domain
        return "general_chat"

    def confidence_decay(self, confidence: float, age_days: float, decay_lambda: Optional[float] = None) -> float:
        lam = self.decay_lambda if decay_lambda is None else float(decay_lambda)
        return max(0.0, min(1.0, float(confidence) * math.exp(-lam * max(0.0, float(age_days)))))

    def decay_memory_confidence(self, metadata: Dict[str, Any], now: Optional[datetime] = None) -> float:
        base = float(metadata.get("confidence", 0.5))
        ts = metadata.get("timestamp") or metadata.get("created_at") or metadata.get("time")
        if not ts:
            return base
        now = now or datetime.now()
        try:
            if isinstance(ts, datetime):
                dt = ts
            else:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00").split("+")[0])
            age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        except Exception:
            age_days = 0.0
        return self.confidence_decay(base, age_days)

    def _prompt_ambiguity(self, text: str) -> float:
        q = str(text or "").strip().lower()
        if not q:
            return 1.0
        ambiguous_terms = ["terbaik", "bagus", "cukup", "sesuai", "itu", "ini", "gimana", "bebas", "optimal", "layak"]
        score = 0.0
        if len(q.split()) < 6:
            score += 0.25
        score += min(0.45, 0.09 * sum(1 for t in ambiguous_terms if t in q))
        if "?" not in q and not any(x in q for x in ["buat", "tambah", "ubah", "jelaskan", "debug", "analisis"]):
            score += 0.15
        return max(0.0, min(1.0, score))

    def _candidate_disagreement(self, candidates: Optional[List[Any]]) -> float:
        candidates = candidates or []
        if len(candidates) < 2:
            return 0.0
        scores = []
        texts = []
        for c in candidates:
            if isinstance(c, dict):
                if "score" in c: scores.append(float(c.get("score", 0.0)))
                if "plan" in c: texts.append(str(c.get("plan", "")))
                elif "text" in c: texts.append(str(c.get("text", "")))
            else:
                if hasattr(c, "score"): scores.append(float(getattr(c, "score")))
                texts.append(str(c))
        score_disagreement = min(1.0, statistics.pstdev(scores) if len(scores) > 1 else 0.0)
        lexical_disagreement = 0.0
        if len(texts) > 1:
            sims = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    a, b = set(re.findall(r"\w+", texts[i].lower())), set(re.findall(r"\w+", texts[j].lower()))
                    sims.append(len(a & b) / max(1, len(a | b)))
            lexical_disagreement = 1.0 - (sum(sims) / max(1, len(sims)))
        return max(0.0, min(1.0, 0.55 * lexical_disagreement + 0.45 * score_disagreement))

    def _evidence_insufficiency(self, state_report: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> float:
        context = context or {}
        evidence_score = context.get("evidence_score")
        if evidence_score is not None:
            return max(0.0, min(1.0, 1.0 - float(evidence_score)))
        retrieval_unc = float(state_report.get("retrieval_uncertainty", context.get("retrieval_uncertainty", 0.5)))
        graph_unc = float(state_report.get("graph_uncertainty", context.get("graph_uncertainty", 0.5)))
        evidence_strength = float(context.get("evidence_strength", 1.0 - retrieval_unc))
        return max(0.0, min(1.0, 0.45 * retrieval_unc + 0.25 * graph_unc + 0.30 * (1.0 - evidence_strength)))

    def _memory_conflict_score(self, state_report: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> float:
        context = context or {}
        explicit = context.get("memory_conflict") or context.get("conflict_score")
        if explicit is not None:
            return max(0.0, min(1.0, float(explicit)))
        contradictions = context.get("contradictions", 0)
        if contradictions:
            return min(1.0, 0.35 + 0.20 * float(contradictions))
        return max(0.0, min(1.0, float(state_report.get("conflict_uncertainty", 0.0))))

    def decompose_uncertainty(self, query: str, state_report: Dict[str, Any], candidates: Optional[List[Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        context = context or {}
        domain = self.infer_domain(query, context)
        domain_risk = self.DOMAIN_RISK_TABLE.get(domain, 0.35)
        evidence_missing = self._evidence_insufficiency(state_report, context)
        memory_conflict = self._memory_conflict_score(state_report, context)
        prompt_amb = self._prompt_ambiguity(query)
        candidate_dis = self._candidate_disagreement(candidates)
        poor_hist = 1.0 - self.domain_calibration.reliability(domain)
        epistemic = max(0.0, min(1.0,
            0.38 * evidence_missing +
            0.25 * memory_conflict +
            0.22 * poor_hist +
            0.15 * float(state_report.get("retrieval_uncertainty", 0.5))
        ))
        aleatoric = max(0.0, min(1.0,
            0.35 * prompt_amb +
            0.30 * candidate_dis +
            0.20 * domain_risk +
            0.15 * float(state_report.get("entropy_uncertainty", state_report.get("semantic_uncertainty", 0.0)))
        ))
        return {"epistemic": epistemic, "aleatoric": aleatoric}

    def attribute_uncertainty(self, query: str, state_report: Dict[str, Any], candidates: Optional[List[Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        domain = self.infer_domain(query, context)
        domain_risk = self.DOMAIN_RISK_TABLE.get(domain, 0.35)
        source_scores = {
            "evidence_insufficient": self._evidence_insufficiency(state_report, context),
            "memory_conflict": self._memory_conflict_score(state_report, context),
            "prompt_ambiguous": self._prompt_ambiguity(query),
            "high_domain_risk": domain_risk,
            "candidate_disagreement": self._candidate_disagreement(candidates),
            "poor_historical_confidence": 1.0 - self.domain_calibration.reliability(domain),
        }
        sorted_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
        main_source = sorted_sources[0][0]
        secondary = [k for k, v in sorted_sources[1:] if v >= 0.45][:3]
        decomp = self.decompose_uncertainty(query, state_report, candidates, context)
        if source_scores["evidence_insufficient"] >= 0.55:
            action = "retrieve_more_evidence"
        elif source_scores["memory_conflict"] >= 0.55 or source_scores["candidate_disagreement"] >= 0.62:
            action = "self_check_or_resolve_conflict"
        elif source_scores["prompt_ambiguous"] >= 0.55:
            action = "ask_clarification"
        elif domain_risk >= 0.85 and float(state_report.get("confidence", 0.5)) < 0.85:
            action = "abstain_or_use_authoritative_tool"
        elif decomp["epistemic"] > decomp["aleatoric"]:
            action = "retrieve_or_use_tool"
        else:
            action = "answer_with_scope_limits"
        return {
            "uncertainty": float(state_report.get("uncertainty", 0.5)),
            "confidence": float(state_report.get("confidence", 0.5)),
            "domain": domain,
            "domain_risk": domain_risk,
            "main_source": main_source,
            "secondary_sources": secondary,
            "source_scores": {k: round(v, 4) for k, v in source_scores.items()},
            "epistemic_uncertainty": round(decomp["epistemic"], 4),
            "aleatoric_uncertainty": round(decomp["aleatoric"], 4),
            "recommended_action": action,
        }

    def assess_query_state(self, query: str, candidates: Optional[List[Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            base = super().assess_query_state(query, candidates)
        except TypeError:
            base = super().assess_query_state(query)
        except Exception:
            base = {"confidence": 0.5, "uncertainty": 0.5}
        if not isinstance(base, dict):
            base = {"confidence": 0.5, "uncertainty": 0.5, "raw": base}
        attribution = self.attribute_uncertainty(query, base, candidates, context)
        epistemic = attribution["epistemic_uncertainty"]
        aleatoric = attribution["aleatoric_uncertainty"]
        combined_uncertainty = max(float(base.get("uncertainty", 0.5)), 0.62 * epistemic + 0.38 * aleatoric)
        domain_reliability = self.domain_calibration.reliability(attribution["domain"])
        calibrated_confidence = max(0.0, min(1.0, float(base.get("confidence", 1.0 - combined_uncertainty)) * (0.75 + 0.25 * domain_reliability)))
        base.update({
            "uncertainty": round(combined_uncertainty, 4),
            "confidence": round(calibrated_confidence, 4),
            "source_attribution": attribution,
            "uncertainty_sources_ranked": sorted(attribution["source_scores"].items(), key=lambda x: x[1], reverse=True),
            "main_uncertainty_source": attribution["main_source"],
            "recommended_action": attribution["recommended_action"],
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "domain": attribution["domain"],
            "domain_risk": attribution["domain_risk"],
        })
        self.uncertainty_attribution_history.append({
            "time": datetime.now().isoformat(),
            "query": str(query)[:500],
            "report": {k: base.get(k) for k in ["uncertainty", "confidence", "main_uncertainty_source", "recommended_action", "domain"]},
        })
        self.uncertainty_attribution_history = self.uncertainty_attribution_history[-500:]
        return base

    def update_calibration(self, predicted_confidence: float, actual_success: float, context: Dict = None):
        context = context or {}
        domain = self.infer_domain(context.get("query") or context.get("task") or "", context)
        self.domain_calibration.add(domain, predicted_confidence, actual_success, context)
        try:
            return super().update_calibration(predicted_confidence, actual_success, context)
        except Exception:
            return {"domain": domain, "recorded": True}

    def domain_calibration_report(self) -> Dict[str, Any]:
        return self.domain_calibration.report()

    def report(self) -> Dict[str, Any]:
        try:
            base = super().report()
        except Exception:
            base = {}
        if not isinstance(base, dict):
            base = {"base_report": base}
        base.update({
            "uncertainty_attribution_recent": self.uncertainty_attribution_history[-5:],
            "domain_calibration": self.domain_calibration_report(),
            "decay_lambda": self.decay_lambda,
        })
        return base

__all__ = [
    "UncertaintyEngine",
    "AdaptiveThresholdState",
    "ThresholdCalibrationSuite",
    "DomainCalibrationLedger",
    "attach_uncertainty_engine",
]


if __name__ == "__main__":
    engine = UncertaintyEngine()
    result = engine.assess_query_state(
        "Should I answer directly or ask for more evidence?",
        context={"domain": "general_chat"},
    )
    print(result)
    print(engine.threshold_diagnostics())
