"""Delta actions as LeRobot processor steps, for data.delta_actions.

    DeltaActionsStep     preprocessor, before normalising: the arm joints of
                         every target in the chunk minus the state the chunk
                         starts from; the grippers stay absolute
                         (constants.DELTA_MASK). Remembers that state.
    AbsoluteActionsStep  postprocessor, after un-normalising: adds the
                         remembered state back, so the client gets absolute targets.

openpi's DeltaActions / AbsoluteActions, as LeRobot steps. LeRobot has its own
pair (RelativeActionsProcessorStep), but it assumes a [B, D] state, which
SmolVLA's [B, 1, D] breaks; these steps take both, and are openarm_smolvla's
own, so the two repos' delta runs match exactly.

The steps are registered with LeRobot, so a checkpoint's processors JSON
names them and loading rebuilds them -- once this module is imported. The
link between the two (the remembered state) is not saved: link_delta_steps
restores it after loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch
from lerobot.lerobot_types import EnvTransition
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import NormalizerProcessorStep
from lerobot.processor import ProcessorStep
from lerobot.processor import ProcessorStepRegistry
from lerobot.processor import UnnormalizerProcessorStep
from lerobot.utils.constants import OBS_STATE

from openarm_act_lerobot import constants as C


@ProcessorStepRegistry.register("openarm_delta_actions")
@dataclass
class DeltaActionsStep(ProcessorStep):
    mask: list[bool] = field(default_factory=lambda: list(C.DELTA_MASK))
    state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def offset(self, actions: torch.Tensor) -> torch.Tensor:
        """The state on the masked joints, shaped to add to `actions` ([B, D] or [B, T, D])."""
        if self.state is None:
            raise RuntimeError("no state yet: the preprocessor has to run before the postprocessor")
        state = self.state[:, -1] if self.state.ndim == 3 else self.state  # the newest observation step
        state = state.reshape(-1, state.shape[-1]).to(device=actions.device, dtype=actions.dtype)
        joints = len(self.mask)
        mask = torch.tensor(self.mask, dtype=actions.dtype, device=actions.device)
        offset = torch.zeros(state.shape[0], actions.shape[-1], dtype=actions.dtype, device=actions.device)
        offset[:, :joints] = state[:, :joints] * mask
        return offset[:, None, :] if actions.ndim == 3 else offset

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None:
            return transition
        self.state = state
        actions = transition.get(TransitionKey.ACTION)
        if actions is None:  # inference: only remember the state
            return transition
        new = transition.copy()
        new[TransitionKey.ACTION] = actions - self.offset(actions)
        return new

    def get_config(self) -> dict[str, Any]:
        return {"mask": list(self.mask)}

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("openarm_absolute_actions")
@dataclass
class AbsoluteActionsStep(ProcessorStep):
    delta: DeltaActionsStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.delta is None:
            raise RuntimeError("AbsoluteActionsStep is not linked to a DeltaActionsStep (see link_delta_steps)")
        actions = transition.get(TransitionKey.ACTION)
        if actions is None:
            return transition
        new = transition.copy()
        new[TransitionKey.ACTION] = actions + self.delta.offset(actions)
        return new

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(self, features):
        return features


def has_delta_steps(preprocessor) -> bool:
    return any(isinstance(s, DeltaActionsStep) for s in preprocessor.steps)


def add_delta_steps(preprocessor, postprocessor) -> None:
    """Delta before normalising, absolute after un-normalising; linked."""
    delta = DeltaActionsStep()
    at = next(i for i, s in enumerate(preprocessor.steps) if isinstance(s, NormalizerProcessorStep))
    preprocessor.steps.insert(at, delta)
    at = next(i for i, s in enumerate(postprocessor.steps) if isinstance(s, UnnormalizerProcessorStep))
    postprocessor.steps.insert(at + 1, AbsoluteActionsStep(delta=delta))


def link_delta_steps(preprocessor, postprocessor) -> None:
    """After loading: point the postprocessor's step at the preprocessor's."""
    delta = next((s for s in preprocessor.steps if isinstance(s, DeltaActionsStep)), None)
    absolute = [s for s in postprocessor.steps if isinstance(s, AbsoluteActionsStep)]
    if (delta is None) != (not absolute):
        raise ValueError("the processors disagree about delta actions: one has the step, the other not")
    for step in absolute:
        step.delta = delta
