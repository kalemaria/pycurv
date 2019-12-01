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

## `pycurv` package
This is the main Python package containing modules, classes and functions used
for the following analyses:

- Estimation of membrane curvature using our several tensor voting-based methods
  based on (Page et al. 2002, Graphical Models) and (Tong and Tang 2005, IEEE
  Transactions on Pattern Analysis and Machine Intelligence), details available
  in the pre-print ([Kalemanov et al. 2019, bioRxiv](https://www.biorxiv.org/content/10.1101/579060v1.full)).
  The workflow consists of the following three main steps:
  1. signed surface generation from a segmentation
  2. surface graph generation and surface cleaning
  3. estimation of normal vectors of the true surface per triangle
  4. principle directions and curvatures estimation per triangle.

  The main parameter of our methods, `radius_hit` (borrowed from Tong and Tang
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

## `pycurv_scripts` package
This package contains Python scripts applying the PyCurv package and
combining different functions into the workflows described above, the main are:

- `curvature_calculation.py` script for membrane curvature calculation workflows
  used in ([Bäuerlein et al. 2017, Cell](https://www.cell.com/fulltext/S0092-8674(17)30934-0))
  and ([Collado et al. 2019, Developmental Cell](https://www.cell.com/developmental-cell/fulltext/S1534-5807(19)30865-2))
- `distances_calculation.py` script for membrane distances and thicknesses
  calculation used in (Collado et al. 2019)
- `ribosome_density_calculation.py` script for ribosome density calculation used
  in (Bäuerlein et al. 2017).

## `scripts_running_mindboggle_and_freesurfer` folder
Python and bash scripts running Mindboggle (Klein et al. 2017, PLoS Computational
Biology) and FreeSurfer (Pienaar et al. 2008, International Journal of Imaging
Systems and Technology) curvature estimation functions and extracting the values
to CSV files.

## `pycurv_testing` package
This package was used for testing our and external curvature estimation
algorithms from VTK (Schroeder et al., 2006, Kitware), FreeSurfer and
Mindboggle. It contains:

- code used to generate synthetic surfaces
- error calculation module
- scripts getting FreeSurfer's and Mindboggle's curvatures and calculating
  errors from their output VTK files.
- integration and unit tests for the main PyCurv workflows and functions
- a collection of plotting functions.
- folders with output of curvature tests, e.g. `test_vector_voting_output`),
  and the test surfaces, e.g. `torus/noise0/torus_rr25_csr10.surface.vtp`.

## `experimental_data_sets` folder
Some experimental data can be found here:

- vesicle: membrane segmentation of a vesicle from a cryo-electron tomogram
  (Bäuerlein et al. 2017)
- vesicle: compartment segmentation of a cortical ER membrane from a cryo-electron
  tomogram (Collado et al. 2019)
- embryo: surfaces of C. elegans embryo cells imaged by confocal light
  microscopy and segmented by LimeSeg (Machado et al., BMC Bioinformatics 2019)
- brain: cortical pial surfaces of both human brain hemispheres imaged by MRI
  and segmented by FreeSurfer, taken from [Mindboggle example data](https://osf.io/8cf5z/).

Output of the following curvature algorithms is included for experimental data
(AVV and SSVV output also includes minimum and maximum principal curvatures
calculated by VTK):

- vesicle: AVV
- ER: AVV, SSVV and Mindboggle
- embryo: AVV
- brain: AVV, Mindboggle, FreeSurfer


# Installing PyCurv
Please note that PyCurv depends on one not publicly available Python package,
pyto (Lučić et al., 2016, PMID: 27742578, DOI: 10.1016/j.jsb.2016.10.004), it
has to be requested from its author, Dr. Vladan Lučić.

## Installation instructions with anaconda
The following instruction were tested on SUSE Linux Enterprise Server 12, but
they should work on other Linux-based systems.

1. Install anaconda with [graph-tool](https://graph-tool.skewed.de/) (Peixoto,
   2014) and its dependencies:
   ```
   targetFold=<your_anaconda_path>
   wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
   bash Anaconda3-2019.10-Linux-x86_64.sh -b -p $targetFold

   export PATH=$targetFold/bin:$PATH

   conda config --set allow_conda_downgrades true

   conda install conda=4.6.14

   conda config --set allow_conda_downgrades true
   conda config --add channels pkgw-forge
   conda config --add channels conda-forge
   conda config --add channels ostrokach-forge

   conda install -c pkgw-forge gtk3
   conda install -c conda-forge pygobject
   conda install -c conda-forge matplotlib
   conda install -c ostrokach-forge graph-tool

   export PATH=$targetFold/bin:$PATH
   ```

   `which python` should output your anaconda `targetFold`.
   You should be able to import `graph_tool` from a `python` or `ipython` shell:
   ```python
   from graph_tool.all import *
   ```

2. Add the path to the pyto package (Lučić et al., 2016, PMID: 27742578,
   DOI: 10.1016/j.jsb.2016.10.004) to PYTHONPATH in bashrc.
   (See https://stackoverflow.com/questions/19917492/how-to-use-pythonpath and
   https://docs.python.org/3.6/tutorial/modules.html)

3. To download the PyCurv package, run from a bash shell:
   ```
   cd [pathToInstallation]  # your destination folder
   git clone https://github.com/kalemaria/pycurv.git
   ```
   The folder `pycurv` should be created, containing the modules and folders
   listed here.

4. Install dependencies from the `setup.py`:
   ```
   cd [pathToInstallation]/pycurv
   python setup.py install
   ```
   You should be able to import `pycurv`, `pycurv_testing` and `pycurv_scripts`
   from a `python` or `ipython` shell.

## Installation instructions without anaconda
The following instruction were tested on Ubuntu 18.04, but the process should be
equivalent for other Ubuntu versions. Ubuntu can be installed for free, also in
a virtual machine on other operating systems (Windows or Mac).
Ubuntu 18.04 has `python3` version 3.6.7 preinstalled.

1. Install [graph-tool](https://graph-tool.skewed.de/) (Peixoto, 2014)
   for Ubuntu according to [instructions](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions#debian-ubuntu),
   `DISTRIBUTION=bionic`, but before running `apt-get update` add the public key:
   ```
   apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25
   ```
   Unfortunately, this installation of the graph-tool package does not work with
   anaconda python.

2. Add the path to the pyto package (Lučić et al., 2016, PMID: 27742578,
   DOI: 10.1016/j.jsb.2016.10.004) to PYTHONPATH in bashrc.
   (See https://stackoverflow.com/questions/19917492/how-to-use-pythonpath and
   https://docs.python.org/3.6/tutorial/modules.html)

3. Install [pip3](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)
   (includes setuptools), [venv](https://docs.python.org/3/library/venv.html)
   (from Python version 3.3 on, recommended from version 3.5 on) in e.g.
   `~/workspace`:
   ```
   python3 -m venv ./venv –system-site-packages
   ```
   and activate:
   ```
   source venv/bin/activate
   ```
   `ipython3` should be present and you should be able to import `graph_tool`:
   ```python
   from graph_tool.all import *
   ```

4. To download the PyCurv package, run from a bash shell:
   ```
   cd [pathToInstallation]  # your destination folder
   git clone https://github.com/kalemaria/pycurv.git
   ```
   The folder `pycurv` should be created, containing the modules and folders
   listed here.

5. Install dependencies from the `setup.py`:
   ```
   cd [pathToInstallation]/pycurv
   python setup.py install
   ```
   You should be able to import `pycurv`, `pycurv_testing` and `pycurv_scripts`
   from a `python` or `ipython` shell.

6. To re-create the environment on another computer or after
   re-installation, freeze the current state of the environment packages:
   ```
   pip freeze > requirements_pycurv.txt
   ```
   To re-create the environment:
   ```
   pip install -r requirements_pycurv.txt
   ```


# Running PyCurv

## Running the tests
To run the integration tests of the curvature workflow on synthetic surfaces,
execute from a bash shell:
```
pytest -q [pathToInstallation]/pycurv/pycurv_testing/test_vector_voting.py
```
To run a specific test, for example `test_sphere_curvatures`, run:
```
pytest -q [pathToInstallation]/pycurv/pycurv_testing/test_vector_voting.py::test_sphere_curvatures
```
If it does not work, try to replace `pytest -q` by `python -m pytest`.

A folder `test_vector_voting_output` containing the test results will be created
inside the current directory.

In the same manner, you can run:

- the integration tests of the distances and thicknesses workflow
  (`test_distances_calculation.py`)
- the unit test of histogram area calculation
  (`test_histogram_area_calculation.py`)
- the unit test for some linear algebra functions (`test_linalg.py`)

## Running the experimental data sets
To run the curvature estimation workflow on the vesicle and ER segmentation in
the `experimental_data_sets` folder, just run in a bash shell:
```
cd [pathToInstallation]/pycurv/pycurv_scripts
python curvature_calculation.py
```
The output will be generated in the respective subfolders of the input,
`vesicle` and `ER`. You can change the parameters in the script.


# Reporting bugs
If you have found a bug or have an issue with the software, please open an issue
[here](https://github.com/kalemaria/pycurv/issues).
