pagmo
=====

[![Build Status](https://img.shields.io/circleci/project/github/esa/pagmo2/master.svg?style=for-the-badge)](https://circleci.com/gh/esa/pagmo2)
[![Build Status](https://img.shields.io/travis/esa/pagmo2/master.svg?logo=travis&style=for-the-badge)](https://travis-ci.org/esa/pagmo2)
[![Build Status](https://img.shields.io/appveyor/ci/ci4esa/pagmo2/master.svg?logo=appveyor&style=for-the-badge)](https://ci.appveyor.com/project/ci4esa/pagmo2)
[![Code Coverage](https://img.shields.io/codecov/c/github/esa/pagmo2.svg?style=for-the-badge)](https://codecov.io/github/esa/pagmo2?branch=master)

[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/pagmo.svg?style=for-the-badge)](https://anaconda.org/conda-forge/pagmo)

[![Join the chat at https://gitter.im/pagmo2/Lobby](https://img.shields.io/badge/gitter-join--chat-green.svg?logo=gitter-white&style=for-the-badge)](https://gitter.im/pagmo2/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02338/status.svg)](https://doi.org/10.21105/joss.02338)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1045337.svg)](https://doi.org/10.5281/zenodo.1045336)

**IMPORTANT NOTICE**: pygmo, the Python bindings for pagmo, have been split off into a separate
project, hosted [here](https://github.com/esa/pygmo2). Please update your bookmarks!

pagmo is a C++ scientific library for massively parallel optimization. It is built around the idea of providing
a unified interface to optimization algorithms and to optimization problems and to make their deployment in
massively parallel environments easy.

If you are using pagmo as part of your research, teaching, or other activities, we would be grateful if you could star
the repository and/or cite our work. For citation purposes, you can use the following BibTex entry, which refers
to the [pagmo paper](https://doi.org/10.21105/joss.02338) in the Journal of Open Source Software:

```bibtex
@article{Biscani2020,
  doi = {10.21105/joss.02338},
  url = {https://doi.org/10.21105/joss.02338},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {53},
  pages = {2338},
  author = {Francesco Biscani and Dario Izzo},
  title = {A parallel global multiobjective framework for optimization: pagmo},
  journal = {Journal of Open Source Software}
}
```

The DOI of the latest version of the software is available at [this link](https://doi.org/10.5281/zenodo.1045336).

The full documentation can be found [here](https://esa.github.io/pagmo2/).

Upgrading from pagmo 1.x.x
==========================

If you were using the old pagmo, have a look here on some technical data on what and why a completely
new API and code was developed: https://github.com/esa/pagmo2/wiki/From-1.x-to-2.x

You will find many tutorials in the documentation, we suggest to skim through them to
realize the differences. The new pagmo (version 2) should be considered (and is) as an entirely different code.
