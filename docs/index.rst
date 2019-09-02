.. PyCurv documentation master file, created by
   sphinx-quickstart on Fri Jul 28 17:36:44 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyCurv's documentation!
==================================

This is a Python package developed for calculation of ribosome density on ER and vesicle membranes and for estimation of membrane curvature.

Why PyCurv?
+++++++++++

Because it can be used for the following two analyses:

- Calculation of particle (e.g. ribosomes) density on cellular membranes using a mask with particle coordinates on the membranes and the membrane mask.

- Estimation of membrane curvature using our implementation of Normal Vector Voting algorithm (Page et al., 2002). The workflow consists of the following three main steps:
    1. signed surface generation
    2. surface cleaning using a graph
    3. curvature calculation using a graph generated from the clean surface.

Installation
++++++++++++

Please note that the following Python packages are required and have to be installed:

- Pyto ImageIO (Lučić et al., 2016, PMID: 27742578 DOI: 10.1016/j.jsb.2016.10.004)

- VTK (http://www.vtk.org)

- graph-tool (Peixoto, 2014; https://git.skewed.de/count0/graph-tool)

The package can be run using Python 2.7 versions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
