import unittest
import numpy as np
from numpy.linalg import solve
from numpy.testing import assert_array_equal

class TestSolve(unittest.TestCase):

  '''
  Tests for the method `solve`, 
  which solve a linear matrix equation, or system of linear scalar equations.
  '''

  def test_example(self):
    '''
    TODO: Explain why this test was written
    '''
    a = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])
    expected = np.array([2., 3.])
    x = solve(a, b)
    assert_array_equal(x, expected)
