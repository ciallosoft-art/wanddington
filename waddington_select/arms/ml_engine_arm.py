"""
MLEngineArm — the C-arm's ML engine on its own (= C − LLM), with a swappable model.

Built exactly as WaddingtonCArm builds its online ranker: the same metadata-only DepMap feature policy,
the same anchor-feature drop, the same leave-one-screen-out rows and x10 in-experiment weighting. Only the
model differs (`waddington_select.ml_models`). Two variants per engine:

    ml_engine_<model>   online: retrained on each round's revealed genes (the C-arm's ML)
    ml_static_<model>   static: the round-0 cross-experiment prior, walked down without retraining

`ml_engine_lightgbm` is the reported engine and the reference every other engine is compared against.
"""

from __future__ import annotations

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .waddington_c_arm import _drop_anchor_default, _get_feature_config


class MLEngineArm(OnlineAdaptiveArm):
    def __init__(self, dataset_name: str, batch_size: int, model: str = "lightgbm",
                 online: bool = True, seed: int = 42) -> None:
        training_csv, extra_feats = _get_feature_config(dataset_name)
        super().__init__(
            dataset_name, batch_size,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
            drop_anchor_feats=_drop_anchor_default(),
            model=model,
            seed=seed,
        )
        self._online_updates = online
        self.name = f"ml_{'engine' if online else 'static'}_{model}"

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        if self._online_updates:
            super().update(round_idx, revealed_new)
        else:
            BaseArm.update(self, round_idx, revealed_new)  # remember what was tested; never retrain


def parse_arm_name(name: str) -> tuple[str, bool] | None:
    """'ml_engine_rf' -> ('rf', True); 'ml_static_rf' -> ('rf', False); anything else -> None."""
    for prefix, online in (("ml_engine_", True), ("ml_static_", False)):
        if name.startswith(prefix):
            return name[len(prefix):], online
    return None
