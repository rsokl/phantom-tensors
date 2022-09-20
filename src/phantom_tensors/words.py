# pyright: strict
"""Convenient definitions of

<Word> = NewType('<Word>', int)

to be used as descriptive axis names in shape types"""
from typing import NewType as _NewType

__all__ = [
    "Axis",
    "Batch",
    "Channel",
    "Embed",
    "Filter",
    "Height",
    "Kernel",
    "Logits",
    "Score",
    "Time",
    "Vocab",
    "Width",
    "Axis1",
    "Axis2",
    "Axis3",
    "Axis4",
    "Axis5",
    "Axis6",
    "Dim1",
    "Dim2",
    "Dim3",
    "Dim4",
    "Dim5",
    "Dim6",
]

Axis = _NewType("Axis", int)
Batch = _NewType("Batch", int)
Channel = _NewType("Channel", int)
Embed = _NewType("Embed", int)
Filter = _NewType("Filter", int)
Height = _NewType("Height", int)
Kernel = _NewType("Kernel", int)
Logits = _NewType("Logits", int)
Score = _NewType("Score", int)
Time = _NewType("Time", int)
Vocab = _NewType("Vocab", int)
Width = _NewType("Width", int)

Axis1 = _NewType("Axis1", int)
Axis2 = _NewType("Axis2", int)
Axis3 = _NewType("Axis3", int)
Axis4 = _NewType("Axis4", int)
Axis5 = _NewType("Axis5", int)
Axis6 = _NewType("Axis6", int)

Dim1 = _NewType("Dim1", int)
Dim2 = _NewType("Dim2", int)
Dim3 = _NewType("Dim3", int)
Dim4 = _NewType("Dim4", int)
Dim5 = _NewType("Dim5", int)
Dim6 = _NewType("Dim6", int)
