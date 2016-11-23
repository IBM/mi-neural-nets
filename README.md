Machine Intelligence Core: Neural Nets
=========================================
Status of Travis Continuous Integration:

[![Build Status](https://travis.ibm.com/tkornut/mi-neural-nets.svg?token=9XHfj7QaSbmFqHsyaQes&branch=master)](https://travis.ibm.com/tkornut/mi-neural-nets)

Description
-----------

A subproject of Machine Intelligence Core framework.

The repository contains solutions and applications related to multi-layer (deep) feed-forward (for now) neural nets.

MIC dependencies
------------
   * MIToolchain - the core of MIC framework.

External dependencies
------------
Additionally it depends on the following external libraries:
   * Boost - library of free (open source) peer-reviewed portable C++ source libraries.
   * Armadillo -  a high quality linear algebra library (matrix maths) for the C++ language, aiming towards a good balance between speed and ease of use.

Applications
------------
   *  --

Installation
------------
```
git clone git@github.ibm.com:tkornut/mi-neural-nets.git
cd mi-neural-nets
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/Documents/workspace/mic/
make -j4 install
```

Maintainer
----------
tkornuta

