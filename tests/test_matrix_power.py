import unittest
import numpy as np
from src.matrix_power import matrix_power, LinAlgError

class TestMatrixPower(unittest.TestCase):

  '''Tests for the method `matrix_power`, which raises a square matrix to the (integer) power `n`'''

  def test_invalid_rank(self):
    '''
    The matrix must be a square matrix, otherwise it must raise an error
    '''
    A = np.array([1])
    with self.assertRaises(LinAlgError):
      matrix_power(A, 1)

  def test_invalid_squareness(self):
    '''
    The matrix must be a square matrix, otherwise it must raise an error
    '''
    A = np.array([[1, 2, 3], [1, 2, 2]])
    with self.assertRaises(LinAlgError):
      matrix_power(A, 1)

  def test_invalid_exponent(self):
    '''
    The exponent `n` (to which the square matrix is raised to) must be
    an integer, otherwise we expect an error to be raised
    '''
    invalid_exp = [-2.3, 4.5, 'a', ' ', [], None]
    A = np.array([[1, 2], [1, 2]])
    for x in invalid_exp:
      with self.assertRaises(TypeError):
        matrix_power(A, x)

  def test_type_object_array(self):
    '''
    `matrix_power` falls back to `np.dot` when the matrix is of type `object`
    '''
    A = np.array([[4, 2], [3, 7]], dtype='object')
    expected = [[22, 22], [33, 55]]
    self.assertTrue(np.all(matrix_power(A, 2) == expected))

  def test_type_array(self):
    '''
    `matrix_power` uses `np.matmul` when the matrix is of any type other than `object`
    '''
    A = np.array([[4, 2], [3, 7]])
    expected = [[22, 22], [33, 55]]
    self.assertTrue(np.all(matrix_power(A, 2) == expected))

  def test_type_stack_objects(self):
    '''
    `matrix_power` does not support stacks of object arrays
    '''
    # TODO: How to create a stack of object arrays?
    pass

  def test_exponent_0(self):
    '''
    When the exponent is 0, `matrix_power` should return the identity matrix
    of the same dimensions
    '''
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    self.assertTrue(np.all(matrix_power(A, 0) == expected))
