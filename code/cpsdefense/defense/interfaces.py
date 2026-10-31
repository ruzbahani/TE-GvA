from typing import Mapping, Protocol

from cpsdefense.core.types import Action, ToolSpec


class StateView(Protocol):
    def version(self) -> int:
        pass

    def last_changed_version(self, subject_id: str) -> int:
        pass

    def exists(self, target_id: str, kind: str) -> bool:
        pass

    def predicate(self, name: str, action: Action) -> bool:
        pass


class Effector(Protocol):
    def sample_duration(self, spec: ToolSpec, action: Action) -> float:
        pass

    def apply(self, action: Action, params: Mapping, now: float) -> dict:
        pass

    def compensate(self, action: Action, now: float) -> dict:
        pass
