import unittest
import numpy as np
from src.matrix_power import matrix_power

class TestMatrixPower(unittest.TestCase):

  '''Tests for the method `matrix_power`, which raises a square matrix to the (integer) power `n`'''

  def test_matrix_squared(self):
    '''
    TODO: Explain why this test was written
    '''
    A = np.array([[1, 2], [1, 2]])
    expected = np.array([[3, 6], [3, 6]])
    self.assertTrue(np.all(matrix_power(A, 2) == expected))
