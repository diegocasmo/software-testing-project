import unittest
import numpy as np
from numpy.linalg import solve, LinAlgError

class TestSolve(unittest.TestCase):

  '''
  Tests for the method `solve`, 
  which solve a linear matrix equation, or system of linear scalar equations.
  '''

  def test_squareness(self):
    '''
    Matrix should be square, otherwise exception should be raised.
    '''
    a = np.array([[1, 2, 3], [1, 2, 0]])
    b = np.array([1, 0])
    with self.assertRaises(LinAlgError):
      solve(a,b)

  def test_rank(self):
    '''
    Matrix should be of rank 2 or higher, otherwise exception should be raised.
    '''
    a = np.array([0])
    b = np.array([1])
    with self.assertRaises(LinAlgError):
      solve(a,b)

  def test_b_zero_dim(self):
    '''
    Test with only b as zero dimentional, should raise Value error
    '''
    a = np.array([[1,2],[3,4]])
    b = np.array([1])
    with self.assertRaises(ValueError):
      solve(a,b)
  
  def test_broadcast_error(self):
    '''
    Test for testing broadcast error
    '''
    a = np.array([[1,2],[3,4]])
    b = np.array([[1,2],[3,4],[5,6]])
    with self.assertRaises(ValueError):
      solve(a,b)

  def test_both_empty(self):
    '''
    Test with empty arrays, should thow exception.
    '''
    a = np.array([])
    b = np.array([])
    with self.assertRaises(LinAlgError):
      solve(a,b)

  def test_return_shape(self):
    '''
    Test that the shape of the solution is the same as b
    '''
    a = np.array([[3,1], [1,2]])
    b = np.array([9,8])
    expected = np.array([2., 3.])
    x = solve(a, b)
    self.assertEquals(b.shape, x.shape)

  def test_singular_matrix(self):
    '''
    Test with singular matrix should raise exception
    '''
    a = np.array([[0, 0], [0, 0]])
    b = np.array([0, 0])
    with self.assertRaises(LinAlgError):
      solve(a, b)
      
  def test_positive_integer(self):
    '''
    Test with all positive integers in matrix
    '''
    a = np.array([[3,1], [1,2]])
    b = np.array([9,8])
    expected = np.array([2., 3.])
    x = solve(a, b)
    self.assertTrue(np.all(expected == x))

  def test_negative_integer(self):
    '''
    Test with a negative integer in matrix system
    '''
    a = np.array([[1,1,1], [0,2,5], [2,5,-1]])
    b = np.array([6,-4,27])
    expected = np.array([5., 3., -2.])
    x = solve(a, b)
    self.assertTrue(np.all(expected == x))

  def test_positive_decimal(self):
    '''
    Test solving system with positive decimals
    '''
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    expected = np.array([2., 3.])
    x = solve(a, b)
    self.assertTrue(np.all(expected == x))

  def test_negative_decimal(self):
    '''
    Test solving system with negative decimals
    '''
    a = np.array([[-3.0, -1.0], [-1.0, -2.0]])
    b = np.array([9.0, 8.0])
    expected = np.array([-2., -3.])
    x = solve(a, b)
    self.assertTrue(np.all(expected == x))
