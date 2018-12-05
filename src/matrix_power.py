import operator

from numpy.core import (asanyarray, matmul, dot, empty_like)
from numpy.lib.twodim_base import eye
from numpy import linalg

# Error object
class LinAlgError(Exception):
    """
    Generic Python-exception-derived object raised by linalg functions.
    General purpose exception class, derived from Python's exception.Exception
    class, programmatically raised in linalg functions when a Linear
    Algebra-related condition would prevent further correct execution of the
    function.
    Parameters
    ----------
    None
    Examples
    --------
    >>> from numpy import linalg as LA
    >>> LA.inv(np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "...linalg.py", line 350,
        in inv return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))
      File "...linalg.py", line 249,
        in solve
        raise LinAlgError('Singular matrix')
    numpy.linalg.LinAlgError: Singular matrix
    """
    pass

def matrix_power(a, n):
    """
    Raise a square matrix to the (integer) power `n`.

    For positive integers `n`, the power is computed by repeated matrix
    squarings and matrix multiplications. If ``n == 0``, the identity matrix
    of the same shape as M is returned. If ``n < 0``, the inverse
    is computed and then raised to the ``abs(n)``.

    .. note:: Stacks of object matrices are not currently supported.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be "powered."
    n : int
        The exponent can be any integer or long integer, positive,
        negative, or zero.

    Returns
    -------
    a**n : (..., M, M) ndarray or matrix object
        The return value is the same shape and type as `M`;
        if the exponent is positive or zero then the type of the
        elements is the same as those of `M`. If the exponent is
        negative the elements are floating-point.

    Raises
    ------
    LinAlgError
        For matrices that are not square or that (for negative powers) cannot
        be inverted numerically.

    Examples
    --------
    >>> from numpy.linalg import matrix_power
    >>> i = np.array([[0, 1], [-1, 0]]) # matrix equiv. of the imaginary unit
    >>> matrix_power(i, 3) # should = -i
    array([[ 0, -1],
           [ 1,  0]])
    >>> matrix_power(i, 0)
    array([[1, 0],
           [0, 1]])
    >>> matrix_power(i, -3) # should = 1/(-i) = i, but w/ f.p. elements
    array([[ 0.,  1.],
           [-1.,  0.]])

    Somewhat more sophisticated example

    >>> q = np.zeros((4, 4))
    >>> q[0:2, 0:2] = -i
    >>> q[2:4, 2:4] = i
    >>> q # one of the three quaternion units not equal to 1
    array([[ 0., -1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.],
           [ 0.,  0., -1.,  0.]])
    >>> matrix_power(q, 2) # = -np.eye(4)
    array([[-1.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0.,  0., -1.]])

    """
    a = asanyarray(a)
    _assertRankAtLeast2(a)
    _assertNdSquareness(a)

    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError("exponent must be an integer")

    # Fall back on dot for object arrays. Object arrays are not supported by
    # the current implementation of matmul using einsum
    if a.dtype != object:
        fmatmul = matmul
    elif a.ndim == 2:
        fmatmul = dot
    else:
        raise NotImplementedError(
            "matrix_power not supported for stacks of object arrays")

    if n == 0:
        a = empty_like(a)
        a[...] = eye(a.shape[-2], dtype=a.dtype)
        return a

    elif n < 0:
        a = linalg.inv(a)
        n = abs(n)

    # short-cuts.
    if n == 1:
        return a

    elif n == 2:
        return fmatmul(a, a)

    elif n == 3:
        return fmatmul(fmatmul(a, a), a)

    # Use binary decomposition to reduce the number of matrix multiplications.
    # Here, we iterate over the bits of n, from LSB to MSB, raise `a` to
    # increasing powers of 2, and multiply into the result as needed.
    z = result = None
    while n > 0:
        z = a if z is None else fmatmul(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else fmatmul(result, z)

    return result

def _assertRankAtLeast2(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)

def _assertNdSquareness(*arrays):
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise LinAlgError('Last 2 dimensions of the array must be square')
