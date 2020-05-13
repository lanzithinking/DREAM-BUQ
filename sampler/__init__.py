"""
Import MCMC samplers
Shiwei Lan @ U of Warwick, 2016
Created July 26, 2016
-------------------------------
Modified Sept. 28, 2019 @ ASU
"""

# geometric infinite-dimensional MCMC's
from .geoinfMC import geoinfMC

# geoinfMC using dolfin (FEniCS 1.6.0/1.5.0)
try:
    from .geoinfMC_dolfin import geoinfMC
except Exception:
    pass

# # geoinfMC using hippylib (FEniCS 1.6.0/1.5.0)
# try:
#     from .geoinfMC_hippy import geoinfMC
# except Exception:
#     pass
