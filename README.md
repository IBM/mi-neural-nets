Machine Intelligence Core: Neural Nets
=========================================

Description
-----------

A subproject of Machine Intelligence Core framework.

The repository contains solutions and applications related to multilayer (deep) feedforward (for now) neural nets.

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
cd mi-rl
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/Documents/workspace/mic/
make -j4 install
```

Maintainer
----------
tkornuta

