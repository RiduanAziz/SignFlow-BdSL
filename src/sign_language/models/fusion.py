from __future__ import annotations

import torch
from torch import nn

from sign_language.landmarks.schema import (
    MODALITY_COORD_OFFSETS,
    TOTAL_DYNAMIC_FEATURES,
    TOTAL_HOLISTIC_COORDINATES,
)


class ModalitySpatialEncoder(nn.Module):
    """Spatial projection and normalization for a single landmark modality."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RobustHolisticFusionClassifier(nn.Module):
    """Proposed Model: Robust Holistic Spatial-Temporal Multimodal Fusion Classifier.

    Architecture:
    1. Modality-specific spatial projections for Left Hand, Right Hand, Pose, and Face.
    2. Dynamic modality-presence detection to prevent missing modalities from corrupting fusion.
    3. Bidirectional temporal sequence modeling per modality stream.
    4. Mask-aware temporal pooling excluding padding frames.
    5. Gated cross-modal fusion that dynamically weights observed and reliable modalities.
    6. Final classification head with dropout and residual connection.
    """

    def __init__(
        self,
        input_dim: int = TOTAL_DYNAMIC_FEATURES,  # 4977 or 1659
        num_classes: int = 100,
        modality_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.modality_dim = modality_dim
        self.hidden_dim = hidden_dim

        # Determine feature multiplier (1 for position only, 3 for pos+vel+acc)
        if input_dim % TOTAL_HOLISTIC_COORDINATES == 0:
            multiplier = input_dim // TOTAL_HOLISTIC_COORDINATES
        else:
            multiplier = 1

        self.multiplier = multiplier

        # Compute modality input dimensions
        lh_in = (MODALITY_COORD_OFFSETS["left_hand"][1] - MODALITY_COORD_OFFSETS["left_hand"][0]) * multiplier
        rh_in = (MODALITY_COORD_OFFSETS["right_hand"][1] - MODALITY_COORD_OFFSETS["right_hand"][0]) * multiplier
        pose_in = (MODALITY_COORD_OFFSETS["pose"][1] - MODALITY_COORD_OFFSETS["pose"][0]) * multiplier
        face_in = (MODALITY_COORD_OFFSETS["face"][1] - MODALITY_COORD_OFFSETS["face"][0]) * multiplier

        # Guard against mismatch if input_dim is a smaller subset
        total_expected = lh_in + rh_in + pose_in + face_in
        if total_expected != input_dim:
            # Fallback: slice proportionally or use general linear projection
            self.use_proportional_slices = False
            self.generic_proj = nn.Linear(input_dim, modality_dim * 4)
        else:
            self.use_proportional_slices = True
            self.lh_encoder = ModalitySpatialEncoder(lh_in, modality_dim, dropout)
            self.rh_encoder = ModalitySpatialEncoder(rh_in, modality_dim, dropout)
            self.pose_encoder = ModalitySpatialEncoder(pose_in, modality_dim, dropout)
            self.face_encoder = ModalitySpatialEncoder(face_in, modality_dim, dropout)

        # Modality temporal GRUs (bidirectional)
        self.lh_temporal = nn.GRU(
            modality_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.rh_temporal = nn.GRU(
            modality_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.pose_temporal = nn.GRU(
            modality_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.face_temporal = nn.GRU(
            modality_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )

        temporal_out_dim = hidden_dim * 2  # Bidirectional

        # Gated fusion layers: estimate relevance/reliability gate per modality
        self.lh_gate = nn.Sequential(nn.Linear(temporal_out_dim + 1, 1), nn.Sigmoid())
        self.rh_gate = nn.Sequential(nn.Linear(temporal_out_dim + 1, 1), nn.Sigmoid())
        self.pose_gate = nn.Sequential(nn.Linear(temporal_out_dim + 1, 1), nn.Sigmoid())
        self.face_gate = nn.Sequential(nn.Linear(temporal_out_dim + 1, 1), nn.Sigmoid())

        # Multimodal fusion network
        fused_in_dim = temporal_out_dim * 4
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        # Output classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def _extract_modality_tensors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_proportional_slices:
            proj = self.generic_proj(x)
            d = self.modality_dim
            return proj[..., 0:d], proj[..., d:2*d], proj[..., 2*d:3*d], proj[..., 3*d:4*d]

        # Extract features according to modality offsets and multiplier
        m = self.multiplier

        # Modality offsets in the pos chunk (0:1659)
        lh_s, lh_e = MODALITY_COORD_OFFSETS["left_hand"]
        rh_s, rh_e = MODALITY_COORD_OFFSETS["right_hand"]
        pose_s, pose_e = MODALITY_COORD_OFFSETS["pose"]
        face_s, face_e = MODALITY_COORD_OFFSETS["face"]

        if m == 1:
            lh_x = x[..., lh_s:lh_e]
            rh_x = x[..., rh_s:rh_e]
            pose_x = x[..., pose_s:pose_e]
            face_x = x[..., face_s:face_e]
        else:
            # Slices across pos, vel, acc
            base = TOTAL_HOLISTIC_COORDINATES
            lh_parts = [x[..., i*base + lh_s : i*base + lh_e] for i in range(m)]
            rh_parts = [x[..., i*base + rh_s : i*base + rh_e] for i in range(m)]
            pose_parts = [x[..., i*base + pose_s : i*base + pose_e] for i in range(m)]
            face_parts = [x[..., i*base + face_s : i*base + face_e] for i in range(m)]
            lh_x = torch.cat(lh_parts, dim=-1)
            rh_x = torch.cat(rh_parts, dim=-1)
            pose_x = torch.cat(pose_parts, dim=-1)
            face_x = torch.cat(face_parts, dim=-1)

        return (
            self.lh_encoder(lh_x),
            self.rh_encoder(rh_x),
            self.pose_encoder(pose_x),
            self.face_encoder(face_x),
        )

    @staticmethod
    def _mask_aware_pool(h: torch.Tensor, temporal_mask: torch.Tensor | None) -> torch.Tensor:
        """Temporal pooling weighted by valid non-padding frames."""
        if temporal_mask is None:
            return h.mean(dim=1)
        weights = temporal_mask.to(h.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (h * weights).sum(dim=1) / denom

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        d = x.shape[-1]

        # Temporal validity mask [B, T]
        temporal_mask = None
        if mask is not None:
            if mask.dim() == 2 and mask.shape[-1] == d:
                mask = mask.unsqueeze(1)
            if mask.dim() == 3:
                temporal_mask = (mask.abs().sum(dim=-1) > 1e-5).float()
            elif mask.dim() == 2:
                temporal_mask = mask.float()

        # 1. Spatial modality projections
        lh_feat, rh_feat, pose_feat, face_feat = self._extract_modality_tensors(x)

        # 2. Modality presence scores: presence in sample [B, 1]
        if mask is not None and mask.dim() == 3:
            base = TOTAL_HOLISTIC_COORDINATES
            mask_coords = mask[..., :base] if mask.shape[-1] >= base else mask
            lh_s, lh_e = MODALITY_COORD_OFFSETS["left_hand"]
            rh_s, rh_e = MODALITY_COORD_OFFSETS["right_hand"]
            pose_s, pose_e = MODALITY_COORD_OFFSETS["pose"]
            face_s, face_e = MODALITY_COORD_OFFSETS["face"]
            lh_presence = (mask_coords[..., lh_s:lh_e].abs().sum(dim=[1, 2], keepdim=True) > 1e-4).float()
            rh_presence = (mask_coords[..., rh_s:rh_e].abs().sum(dim=[1, 2], keepdim=True) > 1e-4).float()
            pose_presence = (mask_coords[..., pose_s:pose_e].abs().sum(dim=[1, 2], keepdim=True) > 1e-4).float()
            face_presence = (mask_coords[..., face_s:face_e].abs().sum(dim=[1, 2], keepdim=True) > 1e-4).float()
        else:
            lh_presence = (lh_feat.abs().sum(dim=[1, 2], keepdim=True) > 1e-4).float()
            rh_presence = (rh_feat.abs().sum(dim=[1, 2], keepdim=True) > 1e-4).float()
            pose_presence = (pose_feat.abs().sum(dim=[1, 2], keepdim=True) > 1e-4).float()
            face_presence = (face_feat.abs().sum(dim=[1, 2], keepdim=True) > 1e-4).float()

        # 3. Temporal recurrent encoding
        lh_h, _ = self.lh_temporal(lh_feat)
        rh_h, _ = self.rh_temporal(rh_feat)
        pose_h, _ = self.pose_temporal(pose_feat)
        face_h, _ = self.face_temporal(face_feat)

        # 4. Mask-aware temporal pooling
        lh_pooled = self._mask_aware_pool(lh_h, temporal_mask)
        rh_pooled = self._mask_aware_pool(rh_h, temporal_mask)
        pose_pooled = self._mask_aware_pool(pose_h, temporal_mask)
        face_pooled = self._mask_aware_pool(face_h, temporal_mask)

        # 5. Modality gating
        lh_g = self.lh_gate(torch.cat([lh_pooled, lh_presence.squeeze(-1)], dim=-1)) * lh_presence.squeeze(-1)
        rh_g = self.rh_gate(torch.cat([rh_pooled, rh_presence.squeeze(-1)], dim=-1)) * rh_presence.squeeze(-1)
        pose_g = self.pose_gate(torch.cat([pose_pooled, pose_presence.squeeze(-1)], dim=-1)) * pose_presence.squeeze(-1)
        face_g = self.face_gate(torch.cat([face_pooled, face_presence.squeeze(-1)], dim=-1)) * face_presence.squeeze(-1)

        # Gated representations
        gated_cat = torch.cat([
            lh_pooled * lh_g,
            rh_pooled * rh_g,
            pose_pooled * pose_g,
            face_pooled * face_g,
        ], dim=-1)

        # 6. Multimodal fusion and classification
        fused = self.fusion_mlp(gated_cat)
        logits = self.classifier(self.dropout(fused))
        return logits


class MultimodalFusionClassifier(nn.Module):
    """Combine pose-like spatial features with motion features before classification."""

    def __init__(
        self,
        spatial_dim: int = 42,
        motion_dim: int = 42,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, spatial: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        if spatial.shape[-1] != self.spatial_proj[0].in_features:
            raise ValueError(f"Expected spatial feature dim {self.spatial_proj[0].in_features}, got {spatial.shape[-1]}")
        if motion.shape[-1] != self.motion_proj[0].in_features:
            raise ValueError(f"Expected motion feature dim {self.motion_proj[0].in_features}, got {motion.shape[-1]}")

        spatial_features = self.spatial_proj(spatial)
        motion_features = self.motion_proj(motion)
        pooled = torch.cat([spatial_features.mean(dim=1), motion_features.mean(dim=1)], dim=-1)
        fused = self.fusion(pooled)
        return self.classifier(fused)

