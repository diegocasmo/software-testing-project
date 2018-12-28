import unittest
import numpy as np
from numpy.linalg import eigvals, LinAlgError

class TestEigvals(unittest.TestCase):

  '''Tests for the method `eigvals`, which computes all eigenvalues of an array.'''

  def test_zeroes(self):
    '''
    Tests that the eigenvalues of a matrix with only zeroes are zeroes.
    '''
    A = np.array([[0,0], [0,0]])
    expected = np.array([0, 0])
    self.assertEqual(np.allclose(eigvals(A), expected),True)

  def test_unitmatrix(self):
    '''
    Tests that the unit matrix has the eigenvalues with value one.
    '''
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected = np.array([1, 1, 1])
    self.assertTrue(np.allclose(eigvals(A), expected))

  def test_dependant_rows(self):
    '''
    Test with two rows being scalar multiples of each other resulting in one eigenvalue being zero.
    '''
    A = np.array([[3, -2, 1], [2, 5, -2], [6, -4, 2]])
    expected = np.array([0., 3., 7.])
    self.assertTrue(np.allclose(eigvals(A), expected))
    
    
  def test_integer(self):
    '''
    Tests a case with only nonzero integer values in the matrix and as eigenvalues.
    '''
    A = np.array([[3, 4, -2], [1, 4, -1], [2, 6, -1]])
    expected = np.array([3,  2, 1])
    self.assertEqual(np.allclose(eigvals(A), expected),True)

    
  def test_double_eigenvalue(self):
    '''
    Tests a case where two eigenvalues are equal but not zero or one.
    '''
    A = np.array([[1, -3, 3], [3, -5, 3], [6, -6, 4]])
    expected = np.array([4.,  -2., -2.])
    self.assertEqual(np.allclose(eigvals(A), expected),True)

  def test_complex_eigvals(self):
    '''
    Tests a case with floating point values in the matrix and complex numbers as eigenvalues.
    '''
    A = np.array([[5,7,-1,1,-3,3],[9,0,-2,3,-5,3],[6,-6,6,-6,2,4],[1,2,3,4,5,6],[6,5,4,3,-1,-1],[4,4,4,-7,-7,-7]])
    expected = np.array([-8.48911+1.66883j, -8.48911-1.66883j, 7.31946+6.9641j, 7.31946-6.9641j, 1.45428, 7.88502])
    self.assertEqual(np.allclose(eigvals(A), expected),True)

  def test_multiple_matrices(self):
    '''
    When the input is an array of multiple matrices, the output should be
    an array containing the eigenvalues of the matrices.
    '''
    A = np.array([ [[1,2,3],[3,-2,1],[-1,4,2]], [[5,4,3],[3,4,1],[1,-2,-2]], [[1,0,0],[0,1,0],[0,0,1]]])
    expected = np.array([[-2.28318599, -0.84812653,  4.13131252],
                         [ 7.88056836,  1.62235272, -2.50292108],
                         [ 1.        ,  1.        ,  1.        ]])
    self.assertEqual(np.allclose(eigvals(A), expected),True)

  def test_error_not_matrix(self):
    '''
    If an array is not a matrix, an exeption should be raised.
    ''' 
    one_row = np.array([1, 2, 3])
    uneven_rows = np.array([[1, 2], [3], [2, 1]])
    uneven_rows2 = np.array([[1, 2, 3], [-1, 2, -3], [1, 2, 3, 4]])
    with self.assertRaises(LinAlgError):
        eigvals(one_row)
    with self.assertRaises(LinAlgError):
        eigvals(uneven_rows)
    with self.assertRaises(LinAlgError):
        eigvals(uneven_rows2)
        
  def test_error_not_square_matrix(self):
    '''
    If a matrix is not a square matrix, an exeption should be raised.
    ''' 
    two_x_three = np.array([[1, 2], [2, 1], [-1, 2]])
    three_x_two = np.array([[1, 2, 3], [3, 2, 1]])
    with self.assertRaises(LinAlgError):
        eigvals(two_x_three)
    with self.assertRaises(LinAlgError):
        eigvals(three_x_two)
