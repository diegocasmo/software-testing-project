import unittest
import numpy as np

from numpy.core import (
    zeros, empty, double, Inf, intp
)

from numpy.testing import (
    assert_almost_equal
)

from src.multi_dot_matrix_chain_order import multi_dot_matrix_chain_order, LinAlgError

class TestMultiDotMatrixchainOrder(unittest.TestCase):
    '''
    Tests for the method _multi_dot_matrix_chain_order
    '''

    def test_zero_loops_order(self):
        '''
        Test for prime path [1,2,3,5] with array of length 1 and no cost return
        '''
        arrays = [np.random.random((30, 35))]
        s_exp = np.array([0])

        s = multi_dot_matrix_chain_order(arrays)
        self.assertEqual(s,s_exp)

    def test_zero_loops_cost(self):
        '''
        Test for prime path [1,2,3,4] with array of length 1 and cost matrix
        '''
        arrays = [np.random.random((30, 35))]
        m_exp = np.array([0.0])

        _, m = multi_dot_matrix_chain_order(arrays, return_costs=True)
        self.assertEqual(m,m_exp)
        
    def test_one_loop_order(self):
        '''
        Test for running all loops only one time and not returning costs, 
        prime path [12,13,14,15,10,11,7,8,2,3,5]
        '''
        arrays = [np.random.random((30, 30)),
                  np.random.random((30, 30))]
        s_expected = np.array([[0,0],[0,0]])
        
        s = multi_dot_matrix_chain_order(arrays)

        assert_almost_equal(np.triu(s[:-1, 1:]),
                            np.triu(s_expected[:-1, 1:]))

    def test_one_loop_cost(self):
        '''
        Test for running all loops only one time and return cost matrix,
        prime path [12,13,14,15,10,11,7,8,2,3,4]
        '''
        arrays = [np.random.random((30, 30)),
                  np.random.random((30, 30))]
        
        m_expected = np.array([[0.,27000.],[0.,0.]])
        
        _, m = multi_dot_matrix_chain_order(arrays, return_costs=True)

        assert_almost_equal(np.triu(m), np.triu(m_expected))

    def test_multiple_loops_order(self):
        '''
        Test for testing only the order, running all loops more than one time
        prime paths [12,13,15,10,12] and [12,13,14,15,10,12]
        '''
        arrays = [np.random.random((30, 35)),
                  np.random.random((35, 15)),
                  np.random.random((15, 5)),
                  np.random.random((5, 10)),
                  np.random.random((10, 20)),
                  np.random.random((20, 25))]
        s_expected = np.array([[0,  1,  1,  3,  3,  3],
                               [0,  0,  2,  3,  3,  3],
                               [0,  0,  0,  3,  3,  3],
                               [0,  0,  0,  0,  4,  5],
                               [0,  0,  0,  0,  0,  5],
                               [0,  0,  0,  0,  0,  0]], dtype=int)
        s_expected -= 1  # Cormen uses 1-based index, python does not.

        s = multi_dot_matrix_chain_order(arrays)

        # Only the upper triangular part (without the diagonal) is interesting.
        assert_almost_equal(np.triu(s[:-1, 1:]),
                            np.triu(s_expected[:-1, 1:]))
    
    def test_multiple_loops_cost(self):
        '''
        Test for testing the cost matrix, running all loops more than one time
        prime paths [12,13,15,10,12] and [12,13,14,15,10,12]
        '''
        arrays = [np.random.random((30, 35)),
                  np.random.random((35, 15)),
                  np.random.random((15, 5)),
                  np.random.random((5, 10)),
                  np.random.random((10, 20)),
                  np.random.random((20, 25))]
        m_expected = np.array([[0., 15750., 7875., 9375., 11875., 15125.],
                               [0.,     0., 2625., 4375.,  7125., 10500.],
                               [0.,     0.,    0.,  750.,  2500.,  5375.],
                               [0.,     0.,    0.,    0.,  1000.,  3500.],
                               [0.,     0.,    0.,    0.,     0.,  5000.],
                               [0.,     0.,    0.,    0.,     0.,     0.]])
        
        _, m = multi_dot_matrix_chain_order(arrays, return_costs=True)

        # Only the upper triangular part (without the diagonal) is interesting.
        assert_almost_equal(np.triu(m), np.triu(m_expected))