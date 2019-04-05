from setuptools import setup

setup(
    name='PySurf',
    version='1.0',
    packages=['pysurf', 'scripts', 'testing'],
    url='',
    license='',
    author='Maria Kalemanov, Antonio Martinez-Sanchez',
    author_email='',
    description='',
    install_requires=["numpy", "scipy", "scikit-image", "pandas", "pytest",
                      "matplotlib", "pathlib2", "vtk", "nibabel", "pathos"]
)
