# Phantom Tensors
> Tensor types with variadic shapes, for any array-based library, that work with both static type and runtime type checkers

This project is currently just a rough prototype!

The goal of this project is to let users write tensor-like types with variadic shapes (via [PEP 646](https://peps.python.org/pep-0646/)) that are amendable to both: static type checking (without a mypy plugin), as well as cross-tensor consistent runtime checking of shapes. This is achieved using relatively minimal hacks (no mypy plugin necessary!). 

> Note that mypy does not support PEP 646 yet, but pyright does. You can run pyright on the following examples to see that they do, indeed type-check as expected! 

> Presently, `torch.Tensor` and `numpy.ndarray` are explicitly supported, but it is trivial to add support for other array-like classes.


`phantom_tensors.parse` validates inputs against types-with-shapes and performs [type narrowing](https://mypy.readthedocs.io/en/latest/type_narrowing.html) so that static type checkers are privy to the newly proven type information about those inputs.


```python
from phantom_tensors import parse
from phantom_tensors.torch import Tensor
from phantom_tensors.numpy import NDArray

import torch as tr
import numpy as np

from typing import NewType
A = NewType("A", int)
B = NewType("B", int)
C = NewType("C", int)


t1, arr, t2 = parse(
    # <- enter dimension-binding context
    (tr.rand(9, 2, 9), Tensor[B, A, B]),  # <-binds A=2 & B=9
    (np.ones((2,)), NDArray[A]),  # <- checks A==2
    (tr.rand(9), Tensor[B]),  # <- checks B==9
);  # <- exit dimension-binding scope 
    #    and statically casts x,y,z to shape-typed Tensors

# static type checkers now see
# t1: Tensor[B, A, B] 
# arr: NDArray[A]
# t2: Tensor[B]

w = parse(tr.rand(78), Tensor[A]);  # <- binds A=78
```

As indicated above, the type-checker sees the shaped-tensor/array types. Additionally, these are subclasses of their rightful parents, so we can pass these to functions typed with vanilla `torch.Tensor` and `numpy.ndarry` annotations, and type checkers will be a-ok with that.

```python
def vanilla_numpy(x: np.ndarray): ...
def vanilla_torch(x: tr.Tensor): ...

vanilla_numpy(arr)  # type checker: OK
vanilla_torch(arr)  # type checker: Error! 
vanilla_torch(t1)  # type checker: OK 
```

**Validation performed by `parse`**

```python
# runtime type checking
>>> parse(1, Tensor[A])
---------------------------------------------------------------------------
ParseError: Expected <class 'torch.Tensor'>, got: <class 'int'>

# dimensionality mismatch
>>> parse(tr.ones(3), Tensor[A, A, A])
---------------------------------------------------------------------------
ParseError: shape-(3,) doesn't match shape-type (A=?, A=?, A=?)

# unsatisfied shape pattern
>>> parse(tr.ones(1, 2), Tensor[A, A])
---------------------------------------------------------------------------
ParseError: shape-(1, 2) doesn't match shape-type (A=1, A=1)

# inconsistent dimension sizes across tensors
>>> x, y = parse(
...     (tr.ones(1, 2), Tensor[A, B]),
...     (tr.ones(4, 1), Tensor[B, A]),
... )

---------------------------------------------------------------------------
ParseError: shape-(4, 1) doesn't match shape-type (B=2, A=1)
```

As indicated above, `parse` is able to compare shapes across multiple tensors by entering into a "dimension-binding scope".
One can enter into this context explicitly:

```python
>>> from phantom_tensors import dim_binding_scope

>>> x = parse(np.zeros((2,)), NDArray[B])  # binds B=2
>>> y = parse(np.zeros((3,)), NDArray[B])  # binds B=3
>>> with dim_binding_scope:
...     x = parse(np.zeros((2,)), NDArray[B])  # binds B=2
...     y = parse(np.zeros((3,)), NDArray[B])  # raises!
---------------------------------------------------------------------------
ParseError: shape-(3,) doesn't match shape-type (B=2,)
```

## Compatibility with Runtime Type Checkers

`parse` is not the only way to perform runtime validation using phantom tensors – they work out of the box with 3rd party runtime type checkers like [beartype](https://github.com/beartype/beartype)! How is this possible?

...We do something tricky here! At, runtime `Tensor[A, B]` actually returns a [phantom type](https://github.com/antonagestam/phantom-types). This means that `isinstance(arr, NDArray[A, B])` is, at runtime, *actually* performing `isinstance(arr, PhantomNDarrayAB)`, which is able to perform the type and shape checks.

Thanks to the ability bind dimensions within a specified context, all `beartype` needs to do is faithfully call `isinstance(...)` within said context and we can have the inputs and ouputs of a phantom-tensor-annotated function get checked!

```python
from phantom_tensors import dim_binding_scope
from beartype import beartype

# @dim_binding_scope:
#   ensures A, B, C consistent across all input/output tensor shapes
#   within scope of function
@dim_binding_scope 
@beartype  # <-- adds isinstance checks on inputs & outputs
def matrix_multiply(x: Tensor[A, B], y: Tensor[B, C]) -> Tensor[A, C]:
    a, b = x.shape
    b, c = y.shape
    return cast(Tensor[A, C], tr.rand(a, c))

@beartype
def needs_vector(x: Tensor[int]): ...


x, y = parse(
    (tr.rand(3, 4), Tensor[A, B]),
    (tr.rand(4, 5), Tensor[B, C]),
)

z = matrix_multiply(x, y)
z  # type revealed: Tensor[A, C]

with pytest.raises(Exception):
    # beartype raises error: Tensor[A, C] doesn't match Tensor[A]
    needs_vector(z)  # <- pyright also raises an error!

with pytest.raises(Exception):
    matrix_multiply(x, x)  # <- pyright also raises an error!
```


**TODO:**
- Handle variable-length annotations: Tensor[A, ...]
- Support some broadcasting?
- Lock down what people can and can't pass to Tensor[<>]. E.g. Tensor[int, int] is OK. Tensor[str] is not.
- Make things thread safe