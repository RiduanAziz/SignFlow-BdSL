from __future__ import annotations

import numpy as np

from sign_language.robustness import NoiseScenario, evaluate_robustness


def test_robustness_evaluation_reports_noise_impact() -> None:
    clean = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    noisy = np.array([1.5, 1.2, 1.1], dtype=np.float32)

    scenario = NoiseScenario(
        name="brightness",
        severity=0.2,
        clean_signal=clean,
        noisy_signal=noisy,
    )

    report = evaluate_robustness([scenario])
    assert report["scenario_count"] == 1
    assert report["scenarios"][0]["name"] == "brightness"
    assert report["scenarios"][0]["severity"] == 0.2
    assert report["scenarios"][0]["mean_absolute_error"] > 0.0
