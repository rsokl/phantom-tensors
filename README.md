# Phantom Tensors
> Tensor types with variadic shapes, for any array-based library, that work with both static and runtime type checkers

<p align="center">
  <a href="https://pypi.python.org/pypi/phantom-tensors">
    <img src="https://img.shields.io/pypi/v/phantom-tensors.svg" alt="PyPI" />
  </a>
  <a>
    <img src="https://img.shields.io/badge/python-3.8%20&#8208;%203.12-blue.svg" alt="Python version support" />
  </a>
</p>


**This project is currently just a rough prototype! Inspired by: [phantom-types](https://github.com/antonagestam/phantom-types)**

The goal of this project is to let users write tensor-like types with variadic shapes (via [PEP 646](https://peps.python.org/pep-0646/)) that are: 
- Amendable to **static type checking (without mypy plugins)**. 
    > E.g., pyright can tell the difference between `Tensor[Batch, Channel]` and `Tensor[Batch, Feature]`
- Useful for performing **runtime checks of tensor types and shapes**. 
    > E.g.,  can validate -- at runtime -- that arrays of types `NDArray[A, B]` and `NDArray[B, A]` indeed have transposed shapes with respect with each other.
- Compatible with *any* array-based library (numpy, pytorch, xarray, cupy, mygrad, etc.)
    > E.g. A function annotated with `x: torch.Tensor` can be passed `phantom_tensors.torch.Tensor[N, B, D]`. It is trivial to write custom phantom-tensor flavored types for any array-based library.

`phantom_tensors.parse` makes it easy to declare shaped tensor types in a way that static type checkers understand, and that are validated at runtime:

```python
from typing import NewType

import numpy as np

from phantom_tensors import parse
from phantom_tensors.numpy import NDArray

A = NewType("A", int)
B = NewType("B", int)

# static: declare that x is of type NDArray[A, B]
#         declare that y is of type NDArray[B, A]
# runtime: check that shapes (2, 3) and (3, 2)
#          match (A, B) and (B, A) pattern across
#          tensors
x, y = parse(
    (np.ones((2, 3)), NDArray[A, B]),
    (np.ones((3, 2)), NDArray[B, A]),
)

x  # static type checker sees: NDArray[A, B]
y  # static type checker sees: NDArray[B, A]

```

Passing inconsistent types to `parse` will result in a runtime validation error.
```python
# Runtime: Raises `ParseError` A=10 and A=2 do not match
z, w = parse(
    (np.ones((10, 3)), NDArray[A, B]),
    (np.ones((3, 2)), NDArray[B, A]),
)
```

These shaped tensor types are amenable to static type checking:

```python
from typing import Any

import numpy as np

from phantom_tensors import parse
from phantom_tensors.numpy import NDArray
from phantom_tensors.alphabet import A, B  # these are just NewType(..., int) types

def func_on_2d(x: NDArray[Any, Any]): ...
def func_on_3d(x: NDArray[Any, Any, Any]): ...
def func_on_any_arr(x: np.ndarray): ...

# runtime: ensures shape of arr_3d matches (A, B, A) patterns
arr_3d = parse(np.ones((3, 5, 3)), NDArray[A, B, A])

func_on_2d(arr_3d)  # static type checker: Error!  # expects 2D arr, got 3D

func_on_3d(arr_3d)  # static type checker: OK
func_on_any_arr(arr_3d)  # static type checker: OK
```


Write easy-to-understand interfaces using common dimension names (or make up your own):

```python
from phantom_tensors.torch import Tensor
from phantom_tensors.words import Batch, Embed, Vocab

def embedder(x: Tensor[Batch, Vocab]) -> Tensor[Batch, Embed]:
    ...
```


Using a runtime type checker, such as [beartype](https://github.com/beartype/beartype) or [typeguard](https://github.com/agronholm/typeguard), in conjunction with `phantom_tensors` means that the typed shape information will be validated at runtime across a function's inputs and outputs, whenever that function is called.

```python
from typing import TypeVar, cast
from typing_extensions import assert_type

import torch as tr
from beartype import beartype

from phantom_tensors import dim_binding_scope, parse
from phantom_tensors.torch import Tensor
from phantom_tensors.alphabet import A, B, C

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@dim_binding_scope
@beartype  # <- adds runtime type checking to function's interfaces
def buggy_matmul(x: Tensor[T1, T2], y: Tensor[T2, T3]) -> Tensor[T1, T3]:
    # This is the wrong operation!
    # Will return shape-(T1, T1) tensor, not (T1, T3)
    out = x @ x.T
    
    # We lie to the static type checker to try to get away with it
    return cast(Tensor[T1, T3], out)

x, y = parse(
    (tr.ones(3, 4), Tensor[A, B]),
    (tr.ones(4, 5), Tensor[B, C]),
)

# At runtime beartype raises:
#   Function should return shape-(A, C) but returned shape-(A, A)
z = buggy_matmul(x, y)  # Runtime validation error!

```

## Installation

```shell
pip install phantom-tensors
```

`typing-extensions` is the only strict dependency. Using features from `phantom_tensors.torch(numpy)` requires that `torch`(`numpy`) is installed too. 

## Some Lower-Level Details and Features

Everything on display here is achieved using relatively minimal hacks (no mypy plugin necessary, no monkeypatching). Presently, `torch.Tensor` and `numpy.ndarray` are explicitly supported by phantom-tensors, but it is trivial to add support for other array-like classes.

> Note that mypy does not support PEP 646 yet, but pyright does. You can run pyright on the following examples to see that they do, indeed type-check as expected! 


### Dimension-Binding Contexts

`phantom_tensors.parse` validates inputs against types-with-shapes and performs [type narrowing](https://mypy.readthedocs.io/en/latest/type_narrowing.html) so that static type checkers are privy to the newly proven type information about those inputs. It performs inter-tensor shape consistency checks within a "dimension-binding context". Tensor-likes that are parsed simultaneously are automatically checked within a common dimension-binding context.


```python
import numpy as np
import torch as tr

from phantom_tensors import parse
from phantom_tensors.alphabet import A, B, C
from phantom_tensors.numpy import NDArray
from phantom_tensors.torch import Tensor

t1, arr, t2 = parse(
    # <- Runtime: enter dimension-binding context
    (tr.rand(9, 2, 9), Tensor[B, A, B]),  # <-binds A=2 & B=9
    (np.ones((2,)), NDArray[A]),  # <- checks A==2
    (tr.rand(9), Tensor[B]),  # <- checks B==9
)  # <- Runtime: exit dimension-binding scope 
   #    Statically: casts t1, arr, t2 to shape-typed Tensors

# static type checkers now see
# t1: Tensor[B, A, B] 
# arr: NDArray[A]
# t2: Tensor[B]

w = parse(tr.rand(78), Tensor[A]);  # <- binds A=78 within this context
```

As indicated above, the type-checker sees the shaped-tensor/array types. Additionally, these are subclasses of their rightful parents, so we can pass these to functions typed with vanilla `torch.Tensor` and `numpy.ndarry` annotations, and type checkers will be a-ok with that.

```python
def vanilla_numpy(x: np.ndarray): ...
def vanilla_torch(x: tr.Tensor): ...

vanilla_numpy(arr)  # type checker: OK
vanilla_torch(arr)  # type checker: Error! 
vanilla_torch(t1)  # type checker: OK 
```

#### Basic forms of runtime validation performed by `parse`

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

To reiterate, `parse` is able to compare shapes across multiple tensors by entering into a "dimension-binding scope".
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

#### Support for `Literal` dimensions:

```python
from typing import Literal as L

from phantom_tensors import parse
from phantom_tensors.torch import Tensor

import torch as tr

parse(tr.zeros(1, 3), Tensor[L[1], L[3]])  # static + runtime: OK
parse(tr.zeros(2, 3), Tensor[L[1], L[3]])  #  # Runtime: ParseError - mismatch at dim 0
```

#### Support for `Literal` dimensions and variadic shapes:

In Python 3.11 you can write shape types like `Tensor[int, *Ts, int]`, where `*Ts` represents 0 or more optional entries between two required dimensions. phantom-tensor supports this "unpack" dimension. In this README we opt for `typing_extensions.Unpack[Ts]` instead of `*Ts` for the sake of backwards compatibility.

```python
from phantom_tensors import parse
from phantom_tensors.torch import Tensor

import torch as tr
from typing_extensions import Unpack as U, TypeVarTuple

Ts = TypeVarTuple("Ts")

# U[Ts] represents an arbitrary number of entries
parse(tr.ones(1, 3), Tensor[int, U[Ts], int)  # static + runtime: OK
parse(tr.ones(1, 0, 0, 0, 3), Tensor[int, U[Ts], int])  # static + runtime: OK

parse(tr.ones(1, ), Tensor[int, U[Ts], int])  # Runtime: Not enough dimensions
```

#### Support for [phantom types](https://github.com/antonagestam/phantom-types):

Supports phatom type dimensions (i.e. `int` subclasses that override `__isinstance__` checks):

```python
from phantom_tensors import parse
from phantom_tensors.torch import Tensor

import torch as tr
from phantom import Phantom

class EvenOnly(int, Phantom, predicate=lambda x: x%2 == 0): ...

parse(tr.ones(1, 0), Tensor[int, EvenOnly])  # static return type: Tensor[int, EvenOnly] 
parse(tr.ones(1, 2), Tensor[int, EvenOnly])  # static return type: Tensor[int, EvenOnly] 
parse(tr.ones(1, 4), Tensor[int, EvenOnly])  # static return type: Tensor[int, EvenOnly] 

parse(tr.ones(1, 3), Tensor[int, EvenOnly])  # runtime: ParseError (3 is not an even number)
```



## Compatibility with Runtime Type Checkers

`parse` is not the only way to perform runtime validation using phantom tensors – they work out of the box with 3rd party runtime type checkers like [beartype](https://github.com/beartype/beartype)! How is this possible?

...We do something tricky here! At, runtime `Tensor[A, B]` actually returns a [phantom type](https://github.com/antonagestam/phantom-types). This means that `isinstance(arr, NDArray[A, B])` is, at runtime, *actually* performing `isinstance(arr, PhantomNDArrayAB)`, which dynamically generated and is able to perform the type and shape checks.

Thanks to the ability to bind dimensions within a specified context, all `beartype` needs to do is faithfully call `isinstance(...)` within said context and we can have the inputs and ouputs of a phantom-tensor-annotated function get checked!

```python
from typing import Any

from beartype import beartype  # type: ignore
import pytest
import torch as tr

from phantom_tensors.alphabet import A, B, C
from phantom_tensors.torch import Tensor
from phantom_tensors import dim_binding_scope, parse

# @dim_binding_scope:
#   ensures A, B, C consistent across all input/output tensor shapes
#   within scope of function
@dim_binding_scope 
@beartype  # <-- adds isinstance checks on inputs & outputs
def matrix_multiply(x: Tensor[A, B], y: Tensor[B, C]) -> Tensor[A, C]:
    a, _ = x.shape
    _, c = y.shape
    return parse(tr.rand(a, c), Tensor[A, C])

@beartype
def needs_vector(x: Tensor[Any]): ...

x, y = parse(
    (tr.rand(3, 4), Tensor[A, B]),
    (tr.rand(4, 5), Tensor[B, C]),
)

z = matrix_multiply(x, y)
z  # type revealed: Tensor[A, C]

with pytest.raises(Exception):
    # beartype raises error: input Tensor[A, C] doesn't match Tensor[A]
    needs_vector(z)  # <- pyright also raises an error!

with pytest.raises(Exception):
    # beartype raises error: inputs Tensor[A, B], Tensor[A, B] don't match signature
    matrix_multiply(x, x)  # <- pyright also raises an error!
```

