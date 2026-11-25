"""Reference checkpoint selection only; no controller or test-set selection.

For target-based runs, call observe_initial on the untrained round-0 model, then
call observe after each COMPLETED benign FL round, starting at round 1.
Pass the complete state_dict (including buffers), not a flattened parameter vector.
Accuracy uses percentage units [0, 100]. A frozen checkpoint never changes.
The caller owns training/control scheduling and checkpoint serialization.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Optional


@dataclass(frozen=True)
class ReferenceRule:
    target_accuracy: Optional[float] = None
    stop_round: Optional[int] = None
    max_overshoot_pp: Optional[float] = None
    allow_crossing_fallback: bool = False
    force_initial_reference: bool = False

    def __post_init__(self):
        if (self.target_accuracy is None) == (self.stop_round is None):
            raise ValueError("Specify exactly one of target_accuracy and stop_round")
        if self.stop_round is not None and (
            type(self.stop_round) is not int or self.stop_round < 1
        ):
            raise ValueError("stop_round must be a positive integer")
        if self.target_accuracy is not None and not (
            math.isfinite(self.target_accuracy) and 0 <= self.target_accuracy <= 100
        ):
            raise ValueError("target_accuracy must be in [0, 100]")
        if self.max_overshoot_pp is not None:
            if self.target_accuracy is None:
                raise ValueError("Overshoot tolerance requires an accuracy target")
            if not math.isfinite(self.max_overshoot_pp) or self.max_overshoot_pp < 0:
                raise ValueError("max_overshoot_pp must be finite and nonnegative")
        if type(self.allow_crossing_fallback) is not bool or type(self.force_initial_reference) is not bool:
            raise ValueError("Reference protocol flags must be boolean")
        if self.allow_crossing_fallback and self.max_overshoot_pp is None:
            raise ValueError("Crossing fallback requires a target tolerance band")


class ReferenceSelector:
    def __init__(self, rule: ReferenceRule, *, selection_split_id: str):
        if not selection_split_id.strip():
            raise ValueError("Record the held-out/auxiliary split identifier")
        self.rule = rule
        self.selection_split_id = selection_split_id
        self._checkpoint = None
        self._last_round = 0
        self._failed = False
        self._previous = None

    @property
    def ready(self):
        return self._checkpoint is not None

    def observe_initial(self, validation_accuracy: float, state_dict: Any) -> bool:
        """Inspect the untrained global model at round 0 for target-based runs.

        This is especially important for random-guess targets (e.g., 10% on
        balanced 10-class datasets), which may already be satisfied before any
        benign FL round is executed.
        """
        if self._failed:
            raise RuntimeError("Selection failed; start a new preregistered run")
        if self.ready:
            return False
        if self.rule.target_accuracy is None:
            return False
        if self._previous is not None or self._last_round != 0:
            raise ValueError("Initial reference observation must occur before round 1")
        if not math.isfinite(validation_accuracy) or not 0 <= validation_accuracy <= 100:
            raise ValueError("validation_accuracy must be finite and in [0, 100]")

        target = self.rule.target_accuracy
        tolerance = self.rule.max_overshoot_pp
        if self.rule.force_initial_reference:
            self._previous = (0, validation_accuracy, deepcopy(state_dict))
            self._checkpoint = {
                "state_dict": deepcopy(state_dict),
                "reference_round": 0,
                "reference_validation_accuracy": float(validation_accuracy),
                "requested_accuracy": target,
                "selection_split_id": self.selection_split_id,
                "selection_rule": "forced_initial_reference",
                "reference_fallback": False,
                "reference_absolute_deviation_pp": abs(float(validation_accuracy) - target),
                "max_overshoot_pp": tolerance,
                "reference_band": None if tolerance is None else [
                    max(0.0, target - tolerance), min(100.0, target + tolerance)
                ],
            }
            return True
        # Round-zero capture is intentionally restricted to an explicitly
        # declared target band. The legacy threshold protocol still begins
        # observing completed benign rounds at round 1.
        self._previous = (0, validation_accuracy, deepcopy(state_dict))
        if tolerance is None:
            return False
        reached = (
            max(0.0, target - tolerance)
            <= validation_accuracy
            <= min(100.0, target + tolerance)
        )
        if not reached:
            return False

        self._checkpoint = {
            "state_dict": deepcopy(state_dict),
            "reference_round": 0,
            "reference_validation_accuracy": float(validation_accuracy),
            "requested_accuracy": target,
            "selection_split_id": self.selection_split_id,
            "selection_rule": "initial_target_band",
            "reference_fallback": False,
            "reference_absolute_deviation_pp": abs(float(validation_accuracy) - target),
            "max_overshoot_pp": tolerance,
            "reference_band": None if tolerance is None else [
                max(0.0, target - tolerance), min(100.0, target + tolerance)
            ],
        }
        return True

    def observe(self, completed_round: int, validation_accuracy: float,
                state_dict: Any) -> bool:
        """Return True on reference capture.

        With a target tolerance, capture only the first checkpoint inside the
        target band. Crossing the whole band does NOT capture by default.
        Nearest-crossing fallback is available only when explicitly enabled.
        """
        if self._failed:
            raise RuntimeError("Selection failed; start a new preregistered run")
        if self.ready:
            return False
        if type(completed_round) is not int or completed_round != self._last_round + 1:
            raise ValueError("Observe every completed round consecutively from 1")
        if not math.isfinite(validation_accuracy) or not 0 <= validation_accuracy <= 100:
            raise ValueError("validation_accuracy must be finite and in [0, 100]")

        target = self.rule.target_accuracy
        tolerance = self.rule.max_overshoot_pp
        chosen = None
        selection_rule = None
        fallback = False

        if target is None:
            if completed_round == self.rule.stop_round:
                chosen = (completed_round, validation_accuracy, deepcopy(state_dict))
                selection_rule = "fixed_round"
        elif tolerance is None:
            if validation_accuracy >= target:
                chosen = (completed_round, validation_accuracy, deepcopy(state_dict))
                selection_rule = "first_crossing"
        else:
            lower = max(0.0, target - tolerance)
            upper = min(100.0, target + tolerance)
            if lower <= validation_accuracy <= upper:
                chosen = (completed_round, validation_accuracy, deepcopy(state_dict))
                selection_rule = "target_band"
            elif validation_accuracy > upper and self.rule.allow_crossing_fallback:
                candidates = [(completed_round, validation_accuracy, deepcopy(state_dict))]
                if self._previous is not None:
                    candidates.append(self._previous)
                chosen = min(candidates, key=lambda item: abs(item[1] - target))
                selection_rule = "nearest_crossing_fallback"
                fallback = True

        if chosen is not None:
            selected_round, selected_accuracy, selected_state = chosen
            self._checkpoint = {
                "state_dict": selected_state,
                "reference_round": selected_round,
                "reference_validation_accuracy": float(selected_accuracy),
                "requested_accuracy": target,
                "selection_split_id": self.selection_split_id,
                "selection_rule": selection_rule,
                "reference_fallback": fallback,
                "reference_absolute_deviation_pp": (
                    None if target is None else abs(float(selected_accuracy) - target)
                ),
                "max_overshoot_pp": tolerance,
                "reference_band": None if target is None or tolerance is None else [
                    max(0.0, target - tolerance), min(100.0, target + tolerance)
                ],
            }

        self._previous = (completed_round, validation_accuracy, deepcopy(state_dict))
        self._last_round = completed_round
        return chosen is not None

    def checkpoint(self):
        if not self.ready:
            raise RuntimeError("No reference captured; do not start control")
        return deepcopy(self._checkpoint)


def objective_stabilization_rounds(post_control_accuracies, *, target_accuracy,
                                   tolerance_pp=2.0, window=10):
    """First 1-based start of a full in-band window, or None (right censored).

    Input starts AFTER the first controlled aggregation. The reference capture
    round and pre-control metrics must not be included. Also returns the round
    at which the window is confirmed. This matches a finite-window OSR, not a
    guarantee that all subsequent rounds remain within the band.
    """
    values = list(post_control_accuracies)
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    if not math.isfinite(target_accuracy) or not 0 <= target_accuracy <= 100:
        raise ValueError("target_accuracy must be in [0, 100]")
    if not math.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError("tolerance_pp must be finite and nonnegative")
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
        raise ValueError("Accuracy trace contains invalid values")
    for start in range(len(values) - window + 1):
        if all(abs(v - target_accuracy) <= tolerance_pp
               for v in values[start:start + window]):
            return {"osr": start + 1, "confirmed_at": start + window}
    return None
