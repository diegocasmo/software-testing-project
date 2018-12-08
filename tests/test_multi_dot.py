import unittest
import numpy as np

class TestMultiDot(unittest.TestCase):

  '''
  Tests for the method `muti_dot`, which computes the dot product of two or more
  arrays in a single function call, while automatically selecting the fastest
  evaluation order.
  '''

  def test_first_row_vector(self):
    '''
    If the first argument is 1-D, it is treated as a row vector
    '''
    A_1d = np.array([1, 1])
    B = np.array([[1, 2], [3, 4]])
    result = np.linalg.multi_dot([A_1d, B])
    expected = [4, 6]
    self.assertTrue(np.all(result == expected))

  def test_last_row_vector(self):
    '''
    If the last argument is 1-D, it is treated as a column vector
    '''
    A = np.array([[1, 2], [3, 4]])
    B_1d = np.array([1, 1])
    result = np.linalg.multi_dot([A, B_1d])
    expected = [3, 7]
    self.assertTrue(np.all(result == expected))

  def test_first_and_last_row_vectors(self):
    '''
    Check that it supports both the first and last arguments as 1-D
    following the conventions established in the tests above
    '''
    A_1d = np.array([1, 1])
    B = np.array([[3, 4], [5, 6]])
    C_1d = np.array([2, 2])
    result = np.linalg.multi_dot([A_1d, B, C_1d])
    expected = [36]
    self.assertTrue(np.all(result == expected))

  def test_too_few_input_arrays(self):
    '''
    When not enough arguments are passed to perform the dot product,
    an error must be raised
    '''
    with self.assertRaises(ValueError):
      np.linalg.multi_dot([])

    with self.assertRaises(ValueError):
      np.linalg.multi_dot([np.random.random((3, 3))])

  def test_basic_setup(self):
    '''
    Check `multi_dot` works as expected with valid inputs
    '''
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
    C = np.array([[10, 9, 8], [7, 6, 5], [4, 3, 2]])
    D = np.array([[4, 3, 2], [7, 6, 5], [10, 9, 8]])
    result = np.linalg.multi_dot([A, B, C, D])
    expected = [[7884, 6696, 5508], [22950, 19494, 16038], [38016, 32292, 26568]]
    self.assertTrue(np.all(result == expected))

  def test_zero_matrix(self):
    '''
    If one of the input matrices elements are all zeros, then the end
    result must be zero
    '''
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    C = np.array([[10, 9, 8], [7, 6, 5], [4, 3, 2]])
    result = np.linalg.multi_dot([A, B, C])
    expected = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    self.assertTrue(np.all(result == expected))

  def test_negative_values(self):
    '''
    Check that `multi_dot` correctly computes the dot product
    when some matrices' elements are negative
    '''
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[-4, -3], [-2, -1]])
    result = np.linalg.multi_dot([A, B])
    expected = [[-8, -5], [-20, -13]]
    self.assertTrue(np.all(result == expected))
