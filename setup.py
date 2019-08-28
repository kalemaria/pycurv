from setuptools import setup

setup(
    name='curvaturia',
    version='1.0',
    packages=['curvaturia', 'scripts', 'testing'],
    url='',
    license='',
    author='Maria Kalemanov, Antonio Martinez-Sanchez',
    author_email='',
    description='',
    install_requires=["numpy", "scipy", "scikit-image<0.15", "pandas", "pytest",
                      "matplotlib", "pathlib2", "vtk", "nibabel",
                      "pathos==0.2.2.1", "networkx==2.2"]
)
