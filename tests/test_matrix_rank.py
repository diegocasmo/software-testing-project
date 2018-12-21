import unittest
import numpy as np
from numpy.linalg import matrix_rank

class TestMatrixRank(unittest.TestCase):

  '''Tests for the method `rank`, which computes the rank of a matrix.'''

  def test_full_rank(self):
    '''
    In a full rank matrix, all rows/columns are linearly independent,
    so the value returned by the `rank` method must be the same as
    the number of rows/columns in the matrix
    '''
    A = [[1,-2,3],[0,-3,3],[1,7,0]]
    self.assertEqual(matrix_rank(A),3)
    self.assertEqual(np.transpose(matrix_rank(A)),3)

  def test_symmetric_full_rank(self):
    '''
    Similar to the test above (`test_full_rank`), but the matrix is
    symmetric, which enables a more efficient method for finding
    singular values
    '''
    I = np.eye(4)
    self.assertEqual(matrix_rank(I, hermitian=True),4)
    self.assertEqual(np.transpose(matrix_rank(I,hermitian=True)),4)

  def test_rank_deficient(self):
    '''
    A matrix is said to be rank deficient if it does not have full rank.
    That is, at least one of its rows/columns is a linear combinations of
    another
    '''
    A = [[1,-2,3],[0,-3,3],[1,1,0]]
    self.assertEqual(matrix_rank(A),2)
    self.assertEqual(np.transpose(matrix_rank(A)),2)

  def test_rank_one(self):
    '''
    A matrix of one dimension has rank 1 unless all of its elements
    are zero
    '''
    self.assertEqual(matrix_rank(np.ones((4,))),1)
    self.assertEqual(np.transpose(matrix_rank(np.ones((4,)))),1)

  def test_rank_zero(self):
    '''
    A matrix of one dimension has rank 0 if all of its elements are zero
    '''
    self.assertEqual(matrix_rank(np.zeros((4,))),0)
    self.assertEqual(np.transpose(matrix_rank(np.zeros((4,)))),0)
