import abc

__all__ = ["CustomInstanceCheck"]


class CustomInstanceCheck(abc.ABCMeta):
    """Used to support custom runtime type checks."""

    def __instancecheck__(self, instance: object) -> bool:
        return self.__instancecheck__(instance)
