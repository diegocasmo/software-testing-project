import unittest
import numpy as np
from numpy.linalg import det

class TestDet(unittest.TestCase):

  '''Tests for the method `det`, which computes the determinant of an array.'''

  def test_example(self):
    '''
    TODO: Explain why this test was written
    '''
    A = np.array([[1, 2], [3, 4]])
    expected = -2.0000000000000004
    self.assertEqual(det(A), expected)
