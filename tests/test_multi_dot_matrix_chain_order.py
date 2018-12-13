import unittest
import numpy as np

from src.multi_dot_matrix_chain_order import multi_dot_matrix_chain_order, LinAlgError

class TestMultiDotMatrixchainOrder(unittest.TestCase):
    '''
    Tests for the method _multi_dot_matrix_chain_order
    '''
    def test_examlpe(self):
        A = np.array([[[1, 2],[3, 4]], [[5, 6], [7, 8]]])
        x = multi_dot_matrix_chain_order(A)
        #print(x)
