# NormalModes

A python module to calculate vibrational frequencies, normal modes, and transform between cartesian and normal mode coordinates. This can be used to generate a grid in normal mode coordinates, on which a potential energy surface can be calculated. The output normal mode grid can serve as input
to your electronic structure code of choice.

A file containing the Hessian of the optimized ground state equalibrium structure is required. Alternatively, the Hessian can be calculated through a wrapper to PySCF.

Atomic units are assumed unless otherwise specfied. The provided Hessian is assumed to be in hartree/cm^2, and vibrational frequencies are output in cm^-1. All output cartesian coordinates are written to file in Angstroem.
