import unittest
import numpy as np
from src.matrix_power import matrix_power, LinAlgError

class TestMatrixPower(unittest.TestCase):

  '''Tests for the method `matrix_power`, which raises a square matrix to the (integer) power `n`'''

  def test_invalid_rank(self):
    '''
    `matrix_powers` requires matrices to have a rank higher than 2, otherwise
    it must raise an error
    '''
    A = np.array([1])
    with self.assertRaises(LinAlgError):
      matrix_power(A, 1)

  def test_invalid_squareness(self):
    '''
    The matrix must have the same number of columns and rows (square), otherwise
    it must raise an error
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

  def test_exponent_zero(self):
    '''
    When the exponent is 0, `matrix_power` should return the identity matrix
    of the same dimensions
    '''
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    self.assertTrue(np.all(matrix_power(A, 0) == expected))

  def test_exponent_negative(self):
    '''
    When the exponent `n` is negative, it should do the matrix multiplication
    with the inverse matrix
    '''
    A = np.array([[1, 2], [2, 3]])
    exp = [-1, -2]
    expected = [
      [[-3, 2], [2, -1]], # When n = -1
      [[13, -8.], [-8, 5]] # When n = -2
    ]
    for i, x in enumerate(exp):
      self.assertTrue(np.all(matrix_power(A, x) == expected[i]))

  def test_small_exponents(self):
    '''
    LinAlg handles exponents `n=1,2,3` differently by using 'shortcuts'. These tests simply
    check whether the correct result is returned after the operation is performed when using
    the exponents `n=1,2,3`.
    '''
    A = np.array([[7, 2], [1, 4]])
    exp = [1, 2, 3]
    expected = [
      [[7, 2], [1, 4]], # When n = 1
      [[51, 22], [11, 18]], # When n = 2
      [[379, 190], [95, 94]] # When n = 3
    ]
    for i, x in enumerate(exp):
      self.assertTrue(np.all(matrix_power(A, x) == expected[i]))

  def test_large_exponents(self):
    '''
    LinAlg handles exponents `n>=4` using binary decomposition to reduce the number of
    multiplications. These tests simply check whether the correct result is returned after
    the operation is performed when using several exponents `n>=4`.
    '''
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    exp = [4, 5, 10, 20]
    expected = [
      [[7560,9288,11016],[17118,21033,24948],[26676,32778,38880]], # When n = 4
      [[121824,149688,177552],[275886,338985,402084],[429948,528282,626616]], # When n = 5
      [[132476037840,162775103256,193074168672],[300005963406,368621393481,437236823556],[467535888972,574467683706,681399478440]], # When n = 10
      [[2754686264399677648,7728919972071978264,-5743590393965272736],[6789354768107336462,-2903799870729760247,5849789564142694660],[-7622720801894556340,4910224360178052858,-1003574551458889560]] # When n = 20
    ]
    for i, x in enumerate(exp):
      self.assertTrue(np.all(matrix_power(A, x) == expected[i]))
