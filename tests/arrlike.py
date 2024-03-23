from typing import Any, Tuple


class ImplementsArray:
    def __init__(self, shape: Tuple[int, ...]) -> None:
        assert all(i >= 0 and isinstance(i, int) for i in shape)
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def __array__(self) -> Any: ...


def arr(*shape: int) -> ImplementsArray:
    return ImplementsArray(shape)
