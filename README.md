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
  used in ([Bäuerlein et al. 2017, Cell](https://doi.org/10.1016/j.cell.2017.08.009))
  and ([Collado et al. 2019, Developmental Cell](https://doi.org/10.1016/j.devcel.2019.10.018))
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

   From the same bash shell, `which python` should output
   `<your_anaconda_path>/bin/python`.

   You should be able to import `graph_tool` from a `python` or `ipython` shell:
   ```python
   from graph_tool.all import *
   ```

   In order that your anaconda python is found every time you open a new
   bash shell, add it to PATH by adding the following line to your `~/.bashrc`:
   ```
   export PATH=<your_anaconda_path>/bin:$PATH
   ```


2. Add the path to the pyto package (Lučić et al., 2016, PMID: 27742578,
   DOI: 10.1016/j.jsb.2016.10.004) to PYTHONPATH in your `~/.bashrc`:
   ```
   export PYTHONPATH=<your_path_to_pyto>:$PYTHONPATH
   ```


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


# Applying PyCurv

To test your PyCurv installation, you can run tests on synthetic surfaces or
workflow scripts on the provided experimental data sets, as explained in the
next subsections.
Then, you can build your own PyCurv curvature estimation workflow, as explained
step-by-step in the "User manual" subsection.
For the full documentation of all modules and functions, please consult
`[pathToInstallation]/pycurv/docs/_build/html/py-modindex.html)`.

## Running the tests
To run the integration tests of the curvature workflow on synthetic surfaces,
execute from a bash shell:
```
pytest -q --disable-pytest-warnings [pathToInstallation]/pycurv/pycurv_testing/test_vector_voting.py
```
To run a specific test, for example `test_sphere_curvatures`, run:
```
pytest -q --disable-pytest-warnings [pathToInstallation]/pycurv/pycurv_testing/test_vector_voting.py::test_sphere_curvatures
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
`vesicle` and `ER`.
You can change the parameters and find more workflow examples in the script.

## User manual
If the tests and the examples above worked for you, now you can learn how to
build your own PyCurv curvature estimation workflow.

### Imports
Fist, import the following:
```
from pycurv import pycurv_io as io
from pycurv import run_gen_surface, THRESH_SIGMA1, TriangleGraph, MAX_DIST_SURF
import numpy as np
from scipy import ndimage
TODO
```

### Parameters
Initialize the following parameters for your run:
```
fold = <your_path_to_input>  # output will be also written there
base_filename = <prefix_for_your_output_files>
pixel_size = <nanometers>  # pixel size of the (underlying) segmentation

# for step 1.a):
seg_file = <your_segmentation_file>  # MRC in this example
# for step 1.a)I.:
label = <membrane_label>
cube_size = <pixels>  # optional, try 3 or 5
# for step 1.a)II.:
filled_label = <lumen_label>
# for step 1.b):
surf_file = <your_surface_file>  # VTP in this example
# for step 2.c):
min_component = <number_triangles>  # default 100, to remove small disconnected
                                    # surface components within this size
TODO
```

### Workflow
1. Generate or load the surface.
a) If the input is a segmentation (here MRC), load it first:
```
seg = io.load_tomo(fold + seg_file)
data_type = seg.dtype
```

I. If the segmentation is not filled (contains only membrane label), generate
the surface using the *membrane segmentation* algorithm.
First, get the membrane segmentation:
```
binary_seg = (seg == label).astype(data_type)
```
Then, generate surface delineating the membrane segmentation:
```
surf = run_gen_surface(binary_seg, fold + base_filename, lbl=1)
```
However, the surface is not always oriented properly, especially if there are
holes in the segmentation.
To close small holes (fitting in the given cube) in the segmentation, run
before `run_gen_surface`:
```
cube = np.ones((cube_size, cube_size, cube_size))
binary_seg = ndimage.binary_closing(
    binary_seg, structure=cube, iterations=1).astype(data_type)
```

II. If the segmentation is filled, generate the surface using the *compartment
segmentation* algorithm.
This is the preferred approach, because the surface is always properly oriented.
As in the previous case, first, get the membrane segmentation:
```
binary_seg = (seg == label).astype(data_type)
```
Second, combine the membrane segmentation with the lumen segmentation into
compartment (filled) segmentation:
```
filled_binary_seg = np.logical_or(
    seg == label, seg == filled_label).astype(data_type)
```
Then, generate isosurface around the slightly smoothed compartment segmentation
and apply the mask of membrane segmentation:
```
surf = run_gen_surface(
    filled_binary_seg, fold + base_filename, lbl=1,
    other_mask=binary_seg, isosurface=True, sg=1, thr=THRESH_SIGMA1)
```
In both cases a) and b), the surface is saved to a VTP file named
`fold + base_filename + ".surface.vtp"`.

b) If the input is a surface (here VTP), omit the above steps and load it:
```
surf = io.load_poly(fold + surf_file)
```

2.a) From the surface, generate a "triangle" graph, with vertices at triangle
centers and edges between neighboring triangles:
```
tg = TriangleGraph()
scale = (pixel_size, pixel_size, pixel_size)
tg.build_graph_from_vtk_surface(surf, scale)
```
b) If the surface has borders, they have grown a bit during the surface
generation (in order to bridge upon small holes) and should be removed:
```
tg.find_vertices_near_border(MAX_DIST_SURF * pixel_size, purge=True)
```
c) You may filter out possibly occurring small disconnected fragments:
```
tg.find_small_connected_components(
    threshold=min_component, purge=True, verbose=True)
```
You can check the number of graph vertices and edges before / after each step:
```
print('The graph has {} vertices and {} edges'.format(
    tg.graph.num_vertices(), tg.graph.num_edges()))
```

3. Then, estimate surface normals at each triangle center using the neighboring
triangles:
```
TODO
```

4. Finally, estimate principle directions and curvatures and calculate different
combined indices using one of the tensor voting-based algorithms:
RVV, AVV (default algorithm) or SSVV:
```
TODO
```

The output is a surface with all the calculated values stored as triangle
properties (VTP).

TODO Add a list of properties name and explanation!

To extract the curvatures into a CSV file, run:
```
extract_curvatures_after_new_workflow(
    fold, base_filename, radius_hit, methods=['VV'], exclude_borders=1
```
Because of the last option, two files will be output: with all values and
excluding those within 1 nm to the surface border.

Finally, you can plot your results in the CSV file, using for example
`[pathToInstallation]/pycurv/pycurv_testing/plotting.py`.

# Reporting bugs
If you have found a bug or have an issue with the software, please open an issue
[here](https://github.com/kalemaria/pycurv/issues).
