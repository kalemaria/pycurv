# curvaturia
This software was developed to analyze the membrane-bound ribosome density and curvature of ER membranes in cryo-electron tomograms.

# ribosome_centers_mapping folder
This folder contains MATLAB scripts used to map ribosome center coordinates obtained by template matching onto a membrane mask from tomogram segmentation. Usage details can be found in the
README file inside this folder.

Please note that TOM MATLAB package (Hrabe et al., 2012; http://www.biochem.mpg.de/tom) is required for reading and writing EM and MRC files and generating a mask from a motive list.
MATLAB/2015b version was used to run these scripts.

# pysurf_compact folder
This is a Python package used for the following two analyses:

- Calculation of ribosome density on ER and vesicle membranes using a mask with ribosome coordinates on the membranes (obtained using the scripts in 'ribosome_centers_mapping' folder) and
the membrane mask.

- Estimation of membrane curvature using our implementation of Normal Vector Voting algorithm (Page et al., 2002). The workflow consists of the following three main steps:
    1. signed surface generation
    2. surface cleaning using a graph
    3. curvature calculation using a graph generated from the clean surface.

Please note that the following Python packages are required and have to be installed:
- Pyto ImageIO (Lučić et al., 2016, PMID: 27742578 DOI: 10.1016/j.jsb.2016.10.004)
- VTK (http://www.vtk.org)
- graph-tool (Peixoto, 2014; https://git.skewed.de/count0/graph-tool)

The package can be run using Python 2.7 versions.

# scripts folder
This folder contains two Python scripts using pysurf_compact package:
- ribosome_density_calculation.py script contains functions for running the ribosome density calculation.
- curvature_calculation.py script contains functions for running the curvature calculation.
