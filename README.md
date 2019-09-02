# PyCurv
This Python-based software was developed mainly to analyse curvature of
membranes in 3D originating from high-resolution, noisy cryo-electron tomograms.
Additionally, the software was also applied to other volumetric data with
segmented structures or directly surface data, e.g. brain or organs from MRI and
cells from confocal light microscopy.
Accepted image data formats are: MRC, EM, VTI, NII.
Accepted surface data formats are: VTP, VTK, STL, PLY.

Moreover, the software can be used to calculate distances between two adjacent
membranes and thicknesses of a membrane organelle.

Furthermore, the software enables to calculate density distribution of
particles mapped on a membrane, e.g. membrane-bound ribosomes.

The software output is mostly in VTP format (triangle-mesh surfaces with
numerical properties like curvatures, distances or densities), which can be
visualized and further analysed in 3D using an external tool,
[ParaView](https://www.paraview.org/).
Also CSV table files for plotting the results are produced and many plotting
functions are included.

## PyCurv package
This is the main Python package containing modules, classes and functions used
for the following analyses:

- Estimation of membrane curvature using our several tensor voting-based methods
  based on (Page et al. 2002, Graphical Models) and (Tong and Tang 2005, IEEE
  Transactions on Pattern Analysis and Machine Intelligence), details available
  in the pre-print (Kalemanov et al. 2019, bioRxiv).
  The workflow consists of the following three main steps:
  1. signed surface generation from a segmentation
  2. surface graph generation and surface cleaning
  3. estimation of normal vectors of the true surface per triangle
  4. principle directions and curvatures estimation per triangle.

  The main parameter of our methods, _RadiusHit_ (borrowed from Tong and Tang
  2005 ) should be set to the radius of the smallest feature of interest on the
  input surface (in the target units, e.g. nanometers). It is used to define a
  geodesic neighborhood of triangles for each central triangle.

  Our method of choice is AVV (augmented vector voting), because it proved to be
  the most robust to noisy and irregularly triangulated surface and to variable
  feature size.

- Calculation of distances between two adjacent membranes and thicknesses of a
  membrane organelle, using the membrane surfaces and outgoing normal vectors
  (estimated as in step iii. in the curvature estimation workflow) from the
  first, flat membrane surface.

- Calculation of ribosome density on ER and vesicle membranes using a mask with
  ribosome coordinates on the membranes and the membrane mask.

## scripts package
This package contains Python scripts applying the PyCurv package and
combining different functions into the workflows described above, the main are:
- curvature_calculation.py script for membrane curvature calculation workflows
  used in (Bäuerlein et al. 2017, Cell) and (Collado et al. 2019, bioRxiv)
- distances_calculation.py script for membrane distances and thicknesses
  calculation used in (Collado et al. 2019, bioRxiv)
- ribosome_density_calculation.py script for ribosome density calculation used
  in (Bäuerlein et al. 2017, Cell).

## testing package
This package contains:
 - code used to generate synthetic test volumes and surfaces for testing our and
   external curvature estimation methods from FreeSurfer (Pienaar et al. 2008,
   International Journal of Imaging Systems and Technology) and Mindboggle
   (Klein et al. 2017, PLoS Computational Biology)
 - error calculation module
 - integration and unit tests for the main PyCurv workflows and functions
 - scripts for running the external software, getting curvatures and calculating
   errors from their output VTK files
 - a collection of plotting functions.
 - folder with output of curvature tests ('test_vector_voting_output'), also
   including the test surfaces, e.g. 'torus/noise0/torus_rr25_csr10.surface.vtp'

## experimental_data_sets folder
Some experimental data can be found here:
- vesicle: unfilled segmentation of a vesicle from a cryo-electron tomogram
- ER: filled segmentation of a ER membrane patch from a cryo-electron tomogram


# Installing PyCurv
Please note that PyCurv depends on one not publicly available Python package,
pyto (Lučić et al., 2016, PMID: 27742578, DOI: 10.1016/j.jsb.2016.10.004), it
has to be requested from its author, Dr. Vladan Lučić.

The code can currently be run using Python 2.7 versions, but since Python 2 will
not longer be supported from the beginning of 2020 and pyto package has been
recently made compatible with Python 3, we plan to upgrade our code to Python 3
soon. Installation instructions are already provided for Python 3, with Python 2
in parentheses.

## Installation instructions for Ubuntu 18.04
The following instruction were tested on Ubuntu 18.04, but the process should be
equivalent for other version and other Linux-based systems. Ubuntu can be
installed for free, also in a virtual machine on other operating systems
(Windows or Mac).

1. Ubuntu 18.04 has python3 version 3.6.7 preinstalled. (Install python2:
   `sudo apt install python-minimal`)

2. Install graph-tool (Peixoto, 2014; https://git.skewed.de/count0/graph-tool)
   for Ubuntu according to [instructions](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions#debian-ubuntu),
   `DISTRIBUTION=bionic`, but before running `apt-get update` add the public key:
   ```
   apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25
   ```
   The graph-tool package does not work with anaconda python.

3. Add the path to the pyto package (Lučić et al., 2016, PMID: 27742578,
   DOI: 10.1016/j.jsb.2016.10.004) to PYTHONPATH in bashrc.
   (See https://stackoverflow.com/questions/19917492/how-to-use-pythonpath and
   https://docs.python.org/3.6/tutorial/modules.html)

4. Install [pip3](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)
   (includes setuptools), [venv](https://docs.python.org/3/library/venv.html) (from
   python version 3.3 on, recommended from version 3.5 on) in e.g. ~/workspace:
   ```
   python3 -m venv ./venv –system-site-packages
   ```
   and activate:
   ```
   source venv/bin/activate
   ```
   ipython3 should be present and graph_tool should be imported:
   ```python
   from graph_tool.all import *
   ```
   (python2: install [virtualenv](https://docs.python-guide.org/dev/virtualenvs/#virtualenvironments-ref)
   in ~/workspace/venv2:
   ```
   virtualenv -p /usr/bin/python2.7 --system-site-packages venv2
   ```
   and activate:
   ```
   source venv2/bin/activate
   ```
   install ipython(2):
   `sudo apt install ipython`)

5. Install dependencies from the setup.py provided in this folder:
   ```
   sudo pythonX setup.py install
   ```
   X=2 or 3 for python2 or 3 and try to import pycurv.

6. To re-create the environment on another computer or after
   re-installation, freeze the current state of the environment packages:
   ```
   pip freeze > requirementsX.txt
   ```
   X=2 or 3 for python2 or 3.
   To re-create the environment:
   ```
   pip install -r requirementsX.txt
   ```


# Running PyCurv

## Running the tests
To run the integration tests of the curvature workflow on synthetic surfaces,
execute from a terminal:
```
pytest -q test_vector_voting.py
```
a folder 'test_vector_voting_output' containing the test results will be created
inside the current directory.

In the same manner, you can run:
- the integration tests of the distances and thicknesses workflow
  (test_distances_calculation.py)
- the unit test of histogram area calculation (test_histogram_area_calculation.py)
- the unit test for some linear algebra functions (test_linalg.py)

## Running the experimental data sets
To run the curvature estimation workflow on the vesicle and ER segmentation in
the 'experimental_data_sets' folder, just run in a terminal from the 'scripts'
folder:
```
python curvature_calculation.py
```
The output will be generated in the respective subfolders of the input,
'vesicle' and 'ER'. You can change the parameters in the script.


# Reporting bugs
If you have found a bug or have an issue with the software, please open an issue
[here](https://github.com/kalemaria/pycurv/issues).
