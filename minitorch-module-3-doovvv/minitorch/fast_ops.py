from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # if len(out_shape)==len(in_shape) and np.all(out_shape==in_shape) and np.all(out_strides == in_strides) :
        #     for i in prange(len(out)):
        #         out[i] = fn(in_storage[i])
        #     return
        """
        由于步长可能不一致，即使是简单版本也不能直接使用a的position进行map
        """
        for i in prange(len(out)):
            big_index = np.zeros(len(out_shape), dtype=np.int32)
            small_index = np.zeros(len(in_shape), dtype=np.int32)
            # 得到大tensor的index
            to_index(i, out_shape, big_index)
            # print(big_index)
            # 得到小tensor的index
            broadcast_index(big_index, out_shape, in_shape, small_index)
            # print(out_shape,in_shape)
            # print(small_index)
            # 将small_index转换为position
            j = index_to_position(small_index, in_strides)
            # 得到对应关系i j,应用fn
            out[index_to_position(big_index,out_strides)] = fn(in_storage[j])
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # if len(out_shape)==len(a_shape) == len(b_shape) and  np.all(out_shape==a_shape) and np.all(a_shape==b_shape) and np.all(out_strides==a_strides) and  np.all(a_strides == b_strides) :
        #     for i in prange(len(out)):
        #         out[i] = fn(a_storage[i], b_storage[i])
        #     return
        """
        由于步长可能不一致，即使是简单版本也不能直接使用a的position进行zip
        """
        for i in prange(len(out)):
            big_index = np.zeros(len(out_shape), dtype=np.int32)
            small_index_a = np.zeros(len(a_shape), dtype=np.int32)
            small_index_b = np.zeros(len(b_shape), dtype=np.int32)
            # 得到大tensor的index
            to_index(i, out_shape, big_index)
            # print(big_index)
            # 得到小tensor的index
            broadcast_index(big_index, out_shape, a_shape, small_index_a)
            broadcast_index(big_index, out_shape, b_shape, small_index_b)
            # print(out_shape,b_shape)
            # print(small_index_b)
            # 将small_index转换为position
            j = index_to_position(small_index_a, a_strides)
            k = index_to_position(small_index_b, b_strides)
            # 得到对应关系i j,应用fn
            out[index_to_position(big_index,out_strides)] = fn(a_storage[j],b_storage[k]) 

        # raise NotImplementedError("Need to implement for Task 3.1")
    # JIT（Just-In-Time）函数是一种 即时编译（JIT Compilation） 技术的实现，它的作用是 在运行时将代码动态编译为机器码
    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        for i in prange(len(out)):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, out_index)
            a_index = out_index.copy()
            for j in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = j
                out[i] = fn(out[i], a_storage[index_to_position(a_index, a_strides)])
        # n = len(out)
        # out_dims = len(a_shape)
        # for i in prange(n):
        #     out_index = np.zeros(out_dims)
        #     to_index(i,out_shape,out_index)
        #     out_idx = index_to_position(out_index,out_strides)

        #     reduce_dim_size = a_shape[reduce_dim]

        #     for j in range(reduce_dim_size):
        #         idx_a = out_index.copy()
        #         idx_a[reduce_dim] = j
        #         pos_a = index_to_position(idx_a, a_strides)
        #         out[out_idx] = fn(out[out_idx],a_storage[pos_a])
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    assert a_shape[-1] == b_shape[-2]
    # TODO: Implement for Task 3.2.
    for i in prange(len(out)):
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        to_index(i, out_shape, out_index)
        #对于连续张量，i等于out_idx
        out_idx = index_to_position(out_index, out_strides)
        for j in range(a_shape[-2]):
            a_index = np.zeros(len(a_shape), dtype=np.int32)
            b_index = np.zeros(len(b_shape), dtype=np.int32)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_index[-2] = j
            a_idx = index_to_position(a_index, a_strides)
            b_index[-2] = j
            b_idx = index_to_position(b_index, b_strides)
            out[out_idx] += a_storage[a_idx] * b_storage[b_idx]  
    # raise NotImplementedError("Need to implement for Task 3.2")


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
