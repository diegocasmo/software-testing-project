# Software Testing Project

Test's for numpy's ``numpy.linalg`` module.

### Installation 
  - Make sure you have installed ``pip`` and your system's Python version is ``2.7.x``
  - Install requirements by running ``pip install -r requirements.txt``

### Running Unit Tests
  - Single module: ``python -m unittest tests.test_matrix_power``
  - Single test (within a module): ``python -m unittest tests.test_matrix_power.TestMatrixPower.test_matrix_squared``
  - All tests: ``python -m unittest discover``

### Running Coverage
  - Run all tests (or any variation from above): ``coverage run --branch -m unittest discover``
  - Display report in command line: ``coverage report -m``
  - Build HTML page with report: ``coverage html``
    - Note: I have ignored the entire ``/htmlcov`` dir from git in order to avoid unnecessary conflicts

### Creating More Tests
  - White-box test
    - Create a test module in ``tests/test_<method_name>.py``
      - Make sure to import the method: ``from src.<module_name> import <method_name>``
    - See [./tests/test_matrix_power.py](./tests/test_matrix_power.py) and [./src/matrix_power.py](./src/matrix_power.py) for a setup example
  - Black-box test
    - Create module for the method which will be tested in ``tests/test_<method_name>.py``
    - Import method from numpy as ``from numpy.linalg import <method_name>``
    - See [./tests/test_det.py](./tests/test_det.py) for a setup example
