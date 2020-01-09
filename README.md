# Physics-informed Deep Learning of Divergence-Free Flows

This repository contains code for reconstruction of divergence-free flows
from the given observations using neural networks trained to satisfy physical
constraints.


## Main code dependencies
 - [Numpy](https://numpy.org)
 - [Scipy](https://scipy.org)
 - [Tensorflow](tensorflow.org)


## Installation

The easiest way to install all required dependencies is using `conda`:

    conda env create --name phidlwind -f environment.yml

or

    conda env create --prefix conda_env -f environment.yml

if you prefer to keep `conda` environment inside the project folder.


## Copyright

© 2019 Dmitry I. Kabanov
