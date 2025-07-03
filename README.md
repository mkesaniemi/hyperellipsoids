This project contains the MATLAB and Python implementations of the ellipsoid-fitting methods described in

"Direct Least Square Fitting of Hyperellipsoids,"
IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(1), 63-76

MATLAB implementation has more functionality than the Python implementation: hyperellipsoidfit.py contains only the new methods HES and SOD.

In the root one can find some test code comparing the outputs of the MATLAB and Python implementations.  
It is worth to note that when writing this README.md, results matched only until 9-dimensional ellipsoids and differed with 10-dimensional data.
Some additional unit tests for the MATLAB code can be found under MATLAB/ directory.

Both directories MATLAB and Python contain a visualization demo of the methods, but the demo in Python directory is currently written in MATLAB though it runs the Python code.
Methods were implemented and tested with Python 3.11 and MATLAB 2025a.
