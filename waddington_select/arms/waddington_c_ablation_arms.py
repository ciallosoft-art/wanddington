"""
Component ablations of the reported C-arm (docs/tier_extension_plan.md, part 丙1).

The older ablation arms (waddington_c_no_memory, _no_ml, _shuffled_names) rebuild the C-arm with the legacy
target-aware router and padded vote; the paper keeps their table as the legacy decomposition. These subclass
the reported WaddingtonCArm and change one thing each, so the fusion weight, feature policy, anchor drop and
named-only vote stay the reported ones:

  waddington_c_nomem       no cross-experiment memory in the LLM's prompt
  waddington_c_static_ml   the ML branch keeps its round-0 leave-one-screen-out prior: no in-screen retraining
  waddington_c_anon_names  the LLM sees anonymous identifiers (GENE_xxxxx) and no features: the reasoning
                           ladder's A0, inside the reported fusion

The fourth ablation, the C-arm without its LLM, is the C-arm's own ML engine (ml_engine_lightgbm) and needs
no arm here.
"""

from __future__ import annotations

from .waddington_c_arm import WaddingtonCArm
from .waddington_c_no_ml_arm import _FrozenOnlineArm
from .waddington_c_shuffled_names_arm import _AnonymousNamesLLMArm


class WaddingtonCNoMemoryFinalArm(WaddingtonCArm):
    """The reported C-arm with no cross-experiment memory."""

    def __init__(self, dataset_name: str, batch_size: int, **kwargs) -> None:
        kwargs.setdefault("name", "waddington_c_nomem")
        kwargs["use_memory"] = False
        super().__init__(dataset_name, batch_size, **kwargs)


class WaddingtonCStaticMLArm(WaddingtonCArm):
    """The reported C-arm whose ML branch never retrains on the screen's revealed genes."""

    def __init__(self, dataset_name: str, batch_size: int, **kwargs) -> None:
        kwargs.setdefault("name", "waddington_c_static_ml")
        super().__init__(dataset_name, batch_size, **kwargs)
        # Same model, features and round-0 prior; update() only records the selections.
        self._online.__class__ = _FrozenOnlineArm


class WaddingtonCAnonNamesArm(WaddingtonCArm):
    """The reported C-arm whose LLM sees anonymous identifiers instead of gene names, and no features."""

    _LLM_CLASS = _AnonymousNamesLLMArm

    def __init__(self, dataset_name: str, batch_size: int, **kwargs) -> None:
        kwargs.setdefault("name", "waddington_c_anon_names")
        kwargs["reason_from_structure"] = False
        super().__init__(dataset_name, batch_size, **kwargs)
