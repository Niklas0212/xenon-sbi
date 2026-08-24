'''
WimPyDD is a modular, object–oriented and customizable Python code
that calculates accurate predictions for the expected rates in WIMP
direct–detection experiments within the framework of
Galilean–invariant non–relativistic effective theory in virtually any
scenario, including inelastic scattering, an arbitrary WIMP spin and a
generic WIMP velocity distribution in the Galactic halo. WimPyDD
exploits the factorization of the three main components that enter in
the calculation of direct detection signals: i) the Wilson
coefficients of the effective theory, that encode the dependence of
the signals on the theoretical parameters; ii) a response function
that depends on the nuclear physics and on the features of the
experimental detector (acceptance, energy resolution, response to
nuclear recoils); iii) a halo function that depends on the WIMP
velocity distribution and that encodes the astrophysical inputs. In
WimPyDD these three components are calculated and stored separately
for later interpolation, combining them together only as the last step
of the signal evaluation procedure. This makes the phenomenological
study of the direct detection scattering rate with WimPyDD transparent
and fast also when the parameter space of the WIMP model has a large
dimensionality.

WimPyDD be downloaded from:

wimpydd.hepforge.org/download/
'''



from __future__ import print_function

from .package import *
