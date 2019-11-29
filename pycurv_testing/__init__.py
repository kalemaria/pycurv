"""
This package is intended for testing PyCurv and data visualization purposes:
- synthetic volumes and surfaces generation
- error calculation and integration testing of our and VTK's membrane curvature
estimation algorithms implemented in the PyCurv package against the algorithms
from the Mindboggle and FreeSurfer packages using the synthetic surfaces
- plotting of the above curvature estimation errors to compare the algorithms
- visualization a geodesic neighborhood on a surface
- integration testing of membrane distances and thicknesses calculation
- unit testing of histogram area calculation used for the choice of the best
neighborhood parameter
- unit testing of some linear algebra functions
"""

from .synthetic_volumes import *
from .synthetic_surfaces import *
from .errors_calculation import *
from .calculate_curvature_errors_of_mindboggle_and_freesurfer import *
from .test_vector_voting import *
from .plotting import *
from .visualization import *
from .test_distances_calculation import *
from .test_histogram_area_calculation import *
from .test_linalg import *
