from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pycurv',
    version='2.0.0',
    author='Maria Kalemanov, Antonio Martinez-Sanchez',
    author_email='kalemanov@biochem.mpg.de',
    description='Reliable estimation of membrane curvature for cryo-electron '
                'tomography',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kalemaria/pycurv',
    packages=find_packages(),
    install_requires=["numpy", "scipy", "scikit-image<0.15", "pandas", "pytest",
                      "matplotlib", "pathlib2", "vtk", "nibabel",
                      "pathos==0.2.2.1", "networkx==2.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 "
        "(LGPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='==3',
)
