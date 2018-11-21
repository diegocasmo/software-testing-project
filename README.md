# Software Testing Project

Test's for numpy's ``numpy.linalg`` module.

### Installation 
  - Make sure you have installed ``pip`` and your system's Python version is ``2.7.x``
  - Install requirements by running ``pip install -r requirements.txt``

### Running Unit Tests
  - Single module: ``python -m unittest tests.test_cholesky``
  - Single test (within a module): ``python -m unittest tests.test_cholesky.TestCholesky.test_true``
  - All tests: ``python -m unittest discover``

### Running Coverage
  - Run all tests (or any variation from above): ``coverage run -m unittest discover``
  - Display report in command line: ``coverage report -m``
  - Build HTML page with report: ``coverage html``
    - Note: I have ignored the entire ``/htmlcov`` dir from git in order to avoid unnecessary conflicts
