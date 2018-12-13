import unittest
import numpy as np

from numpy.core import (
    zeros
)

from numpy.testing import (
    assert_almost_equal
)

from src.multi_dot_matrix_chain_order import multi_dot_matrix_chain_order, LinAlgError

class TestMultiDotMatrixchainOrder(unittest.TestCase):
    '''
    Tests for the method _multi_dot_matrix_chain_order
    '''
    # def test_example_from_numpy(self):
    #     a = zeros((10,100))
    #     b = zeros((100,5))
    #     c = zeros((5,50))
    #     d = zeros((10,50))
    #     arrays = np.array([a,b,c,d])
    #     order = multi_dot_matrix_chain_order(arrays)
    #     #print(order)

    def test_dynamic_programming_logic(self):
        # Totally stole this from numpy.linalg test!
        # Test for the dynamic programming part
        # This test is directly taken from Cormen page 376.
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
        s_expected = np.array([[0,  1,  1,  3,  3,  3],
                               [0,  0,  2,  3,  3,  3],
                               [0,  0,  0,  3,  3,  3],
                               [0,  0,  0,  0,  4,  5],
                               [0,  0,  0,  0,  0,  5],
                               [0,  0,  0,  0,  0,  0]], dtype=int)
        s_expected -= 1  # Cormen uses 1-based index, python does not.

        s, m = multi_dot_matrix_chain_order(arrays, return_costs=True)

        # Only the upper triangular part (without the diagonal) is interesting.
        assert_almost_equal(np.triu(s[:-1, 1:]),
                            np.triu(s_expected[:-1, 1:]))
        assert_almost_equal(np.triu(m), np.triu(m_expected))
