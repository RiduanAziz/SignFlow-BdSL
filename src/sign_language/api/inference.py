from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sign_language.config import load_experiment_config
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.models import build_model
from sign_language.utils.device import get_device

logger = logging.getLogger(__name__)


def create_prediction_payload(sequence: np.ndarray) -> dict[str, int | float | str]:
    """Prepare a prediction payload with fixed feature sizing metadata."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Prediction sequence must be a 2D array of shape [time_steps, features].")
    return {
        "sequence_length": int(arr.shape[0]),
        "feature_dim": int(arr.shape[1]),
        "dtype": str(arr.dtype),
    }


class RealTimePredictor:
    """Predictor that executes a validated PyTorch checkpoint for inference."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = get_device(device)
        self.config = load_experiment_config(config_path or "configs/base.yaml")

        self.class_names: list[str] = list(self.config.dataset.labels)
        self.sequence_length: int = self.config.dataset.sequence_length
        self.input_dim: int = self.config.model.input_dim

        # Load checkpoint strictly if provided
        if model_path is not None:
            ckpt_path = Path(model_path)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Required checkpoint file does not exist: {ckpt_path}. "
                    "Refusing to continue with untrained weights."
                )

            ckpt = torch.load(ckpt_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                if "config" in ckpt and isinstance(ckpt["config"], dict):
                    saved_cfg = ckpt["config"]
                    if "model" in saved_cfg and isinstance(saved_cfg["model"], dict):
                        for k, v in saved_cfg["model"].items():
                            setattr(self.config.model, k, v)
                    if "dataset" in saved_cfg and isinstance(saved_cfg["dataset"], dict):
                        for k, v in saved_cfg["dataset"].items():
                            setattr(self.config.dataset, k, v)
                if ckpt.get("class_names"):
                    self.class_names = list(ckpt["class_names"])
                if "input_dim" in ckpt:
                    self.input_dim = ckpt["input_dim"]
                if "sequence_length" in ckpt:
                    self.sequence_length = ckpt["sequence_length"]
                if "model_name" in ckpt:
                    self.config.model.name = ckpt["model_name"]
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                raise ValueError(f"Unrecognized checkpoint format in {ckpt_path}")

            self.config.model.input_dim = self.input_dim
            self.config.model.num_classes = len(self.class_names)
            self.model = build_model(self.config)
            self.model.load_state_dict(state_dict)
        else:
            # If no model_path is provided, build model from config
            self.config.model.num_classes = len(self.class_names)
            self.model = build_model(self.config)

        self.checkpoint_path = str(ckpt_path) if model_path is not None else None
        self.model.to(self.device)
        self.model.eval()
        self.num_classes = len(self.class_names)

        # Preprocessing pipeline
        raw_dim = self.input_dim // 3 if self.input_dim >= 4977 else self.input_dim
        self.pipeline = LandmarkPipeline(feature_dim=raw_dim, sequence_length=self.sequence_length)

        # Canonical BdSL vocabulary support
        self.bdsl_vocab = None
        try:
            from sign_language.config.vocabulary import get_bdsl_vocabulary
            vocab = get_bdsl_vocabulary()
            if vocab.num_classes == self.num_classes:
                self.bdsl_vocab = vocab
        except Exception as exc:
            logger.debug("BdSL vocabulary not available or mismatched: %s", exc)

        # Optional calibration prior artifact matching the model checkpoint
        self.prior_logits = None
        if self.checkpoint_path:
            prior_cand = Path(self.checkpoint_path).with_suffix(".prior.npy")
            if prior_cand.is_file():
                try:
                    vec = np.load(prior_cand)
                    if vec.shape[0] == self.num_classes:
                        self.prior_logits = vec.astype(np.float32)
                except Exception as exc:
                    logger.debug("Could not load calibration prior: %s", exc)

    def predict_logits(
        self,
        sequence: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute raw output logits for a sequence."""
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
            if mask is not None:
                mask = np.asarray(mask, dtype=np.float32)[np.newaxis, :]

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D sequence [T, D], got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input sequence contains non-finite values (NaN or Inf)")

        mask_arr = np.asarray(mask, dtype=np.float32) if mask is not None else None

        # Normalize and resample if needed
        if arr.shape[1] == self.pipeline.feature_dim:
            arr, mask_arr = self.pipeline.normalize_sequence_with_mask(
                arr, mask=mask_arr, include_dynamics=(self.input_dim >= 4977)
            )

        if arr.shape[1] != self.input_dim:
            if arr.shape[1] > self.input_dim:
                # Slicing extra dimensions (e.g. 4977 dynamic features passed to 1659 spatial model)
                arr = arr[:, :self.input_dim]
                if mask_arr is not None and mask_arr.shape[1] > self.input_dim:
                    mask_arr = mask_arr[:, :self.input_dim]
            else:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.input_dim}, got {arr.shape[1]}"
                )

        with torch.no_grad():
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
            m = (
                torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                if mask_arr is not None
                else None
            )
            logits = self.model(x, mask=m)
            return logits.squeeze(0).cpu().numpy()

    def predict(self, sequence: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Backward-compatible predict method returning raw output logits."""
        return self.predict_logits(sequence, mask=mask)

    def predict_with_logits(
        self,
        sequence: np.ndarray,
        mask: np.ndarray | None = None,
        confidence_threshold: float = 0.0,
        calibrate: bool = True,
        calibration_alpha: float = 0.85,
    ) -> dict[str, Any]:
        """Single-forward-pass inference producing raw logits, probabilities, and class metadata."""
        logits = self.predict_logits(sequence, mask=mask)

        # Apply prior debiasing if available to prevent degenerate class saturation
        eval_logits = logits.copy()
        if calibrate and self.prior_logits is not None:
            eval_logits = eval_logits - (calibration_alpha * self.prior_logits)

        exp_logits = np.exp(eval_logits - np.max(eval_logits))
        probs = exp_logits / np.sum(exp_logits)

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class_{pred_idx}"

        meets_threshold = confidence >= confidence_threshold

        bengali_label = pred_label
        english_gloss = pred_label
        transliteration = ""
        if self.bdsl_vocab is not None:
            try:
                if pred_idx < self.bdsl_vocab.num_classes:
                    item = self.bdsl_vocab.get_by_id(pred_idx)
                else:
                    item_id = self.bdsl_vocab.get_id_by_label(pred_label)
                    item = self.bdsl_vocab.get_by_id(item_id)
                bengali_label = item.get("bengali_label", pred_label)
                english_gloss = item.get("english_gloss", pred_label)
                transliteration = item.get("transliteration", "")
            except Exception:
                pass

        top_indices = np.argsort(probs)[::-1][:min(5, len(probs))]
        top_k = []
        for i in top_indices:
            idx = int(i)
            lbl = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
            b_lbl = lbl
            e_lbl = lbl
            t_lbl = ""
            if self.bdsl_vocab is not None:
                try:
                    if idx < self.bdsl_vocab.num_classes:
                        vitem = self.bdsl_vocab.get_by_id(idx)
                    else:
                        vitem_id = self.bdsl_vocab.get_id_by_label(lbl)
                        vitem = self.bdsl_vocab.get_by_id(vitem_id)
                    b_lbl = vitem.get("bengali_label", lbl)
                    e_lbl = vitem.get("english_gloss", lbl)
                    t_lbl = vitem.get("transliteration", "")
                except Exception:
                    pass
            top_k.append({
                "class_index": idx,
                "label": lbl,
                "bengali_label": b_lbl,
                "english_gloss": e_lbl,
                "transliteration": t_lbl,
                "confidence": float(probs[idx]),
            })

        return {
            "predicted_class": pred_idx,
            "predicted_label": pred_label if meets_threshold else "Unknown (Low Confidence)",
            "bengali_label": bengali_label if meets_threshold else "অজানা (স্বল্প আত্মবিশ্বাস)",
            "english_gloss": english_gloss if meets_threshold else "Unknown",
            "transliteration": transliteration if meets_threshold else "",
            "confidence": confidence,
            "meets_threshold": meets_threshold,
            "probabilities": probs.tolist(),
            "top_k": top_k,
            "logits": [float(v) for v in logits],
        }

    def predict_label(
        self,
        sequence: np.ndarray,
        mask: np.ndarray | None = None,
        confidence_threshold: float = 0.0,
        calibrate: bool = True,
        calibration_alpha: float = 0.85,
    ) -> dict[str, Any]:
        """Predict class label, confidence score, top-k candidates, and probabilities."""
        return self.predict_with_logits(
            sequence=sequence,
            mask=mask,
            confidence_threshold=confidence_threshold,
            calibrate=calibrate,
            calibration_alpha=calibration_alpha,
        )
