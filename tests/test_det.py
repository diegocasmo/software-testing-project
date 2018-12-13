import unittest
import numpy as np
from numpy.linalg import det, LinAlgError

class TestDet(unittest.TestCase):

  '''Tests for the method `det`, which computes the determinant of a matrix.'''

  def test_integer_positive(self):
    '''
    Tests the case where the matrix contains only positive integers
    '''
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[7, 7, 7], [3, 4, 3], [1, 5, 1]])
    expected_A = -2.0
    expected_B = 0.0
    self.assertAlmostEqual(det(A), expected_A)    
    self.assertAlmostEqual(det(B), expected_B)

    
  def test_integer_negative(self):
    '''
    Tests the case where the matrix contains some negative integers
    '''
    A = np.array([[-1, -2], [-3, -4]])
    B = np.array([[-7, -7, -7], [-3, -4, 3], [1, -5, -1]])
    expected_A = -2.0
    expected_B = -266.0
    self.assertAlmostEqual(det(A), expected_A)    
    self.assertAlmostEqual(det(B), expected_B)

        
  def test_decimal_positive(self):
    '''
    Tests the case where the matrix contains only positive decimal numbers
    '''
    A = np.array([[0.5, 0.6], [0.3, 0.9]])
    expected = 0.27
    self.assertEqual(det(A), expected)
 
    
  def test_decimal_negative(self):
    '''
    Tests the case where the matrix contains some negative decimal numbers
    '''
    A = np.array([[-0.5, -0.6], [0.3, -0.9]])
    expected = 0.63
    self.assertEqual(det(A), expected)


  def test_single_element(self): 
    '''
    For 1 dimensional matrices, the element in the matrix should 
    be returned
    '''
    A = np.array([[1]])
    B = np.array([[0.33]])
    expected_A = 1.0
    expected_B = 0.33
    self.assertEqual(det(A), expected_A)
    self.assertEqual(det(B), expected_B)


  def test_zeros(self):
    '''
    When a matrix contains on or more rows where all elements are 0 the 
    determinant will be 0
    '''
    A = np.array([[0, 0, 0], [0, 0, 0] , [0, 0, 0]])
    B = np.array([[9, 6, 3], [33, 700, 5] , [0, 0, 0]])
    expected = 0.0
    self.assertEqual(det(A), expected)
    self.assertEqual(det(B), expected)


  def test_dependant_rows(self): 
    '''
    When a matrix contains rows which are scalar multiples of eachother the 
    determinant should be 0
    '''
    A = np.array([[1, 2, 3], [2, 4, 6] , [4, 8, 12]])
    B = np.array([[1, 1, 1, 1], [99, 99, 99, 99] , 
                  [4, 8, 12, 70], [3, 6, 1, 5]])
    expected = 0.0
    self.assertEqual(det(A), expected)
    self.assertAlmostEqual(det(B), expected)


  def test_empty(self):
    '''
    Empty arrays should raise an exeption
    '''
    A = np.array([])
    B = np.array([[]])
    with self.assertRaises(LinAlgError):
        det(A)
    with self.assertRaises(LinAlgError):
        det(B)  

        
  def test_proper_matrix(self):
    '''
    If an array is not a matrix an exeption should be raised
    ''' 
    A = np.array([1, 2, 3])
    B = np.array([[1, 2], [3], [4, 5]])
    C = np.array([[3, 3, 3], [3, 3, 3], [4, 4, 4, 4]])
    with self.assertRaises(LinAlgError):
        det(A)
    with self.assertRaises(LinAlgError):
        det(B)
    with self.assertRaises(LinAlgError):
        det(C)

        
  def test_squareness(self):
    '''
    If a matrix is not square an exeption should be raised
    ''' 
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([[3, 3, 3], [3, 3, 3]])
    with self.assertRaises(LinAlgError):
        det(A)
    with self.assertRaises(LinAlgError):
        det(B)
    

  def test_multiple_matrices(self):
    '''
    When the input is an array of matrices, the output should be
    an array containing the determinants of those matrices
    ''' 
    A = np.array([ [[1, 2], [3, 4]], 
                   [[1, 2], [2, 19]], 
                   [[1, 3], [3, 1]] ])
    
    B = np.array([ [[7, 5, 11], [12, 6, 3], [8, 8, 8]], 
                   [[1, 2, 9], [2, 1, 19], [1, 3, 0]], 
                   [[7, 7, 7], [3, 4, 3], [1, 5, 1]] ])

    expected_A = np.array([-2., 15., -8.])
    expected_B = np.array([336., 26., 0.])
    
    self.assertTrue(np.allclose(det(A),expected_A))
    self.assertTrue(np.allclose(det(B),expected_B))

   
if __name__=='__main__':
    unittest.main()