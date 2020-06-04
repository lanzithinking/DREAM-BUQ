# This is to test setting Function values in parallel
# https://fenicsproject.org/qa/10005/setting-function-values-in-parallel-repeated-question/

import dolfin as df
import sys
import numpy as np

n_cells = 10;#int(sys.argv[1])

mesh = df.UnitSquareMesh(n_cells, n_cells)                                                   
V = df.FunctionSpace(mesh, 'CG', 2)  
u = df.Function(V)                                                                 

vec = u.vector()
values = vec.get_local()                                            

dofmap = V.dofmap()                                                             
my_first, my_last = dofmap.ownership_range()                # global

# 'Handle' API change of tabulate coordinates
if df.__version__ >= '1.6.0':
    x = V.tabulate_dof_coordinates().reshape((-1, 2))
else:
    x = V.dofmap().tabulate_all_coordinates(mesh)

unowned = dofmap.local_to_global_unowned()
dofs = filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned, 
              range(my_last-my_first))

dof_local=np.array(list(dofs))
# import pydevd; pydevd.settrace() # parallel debug in pydev
x = x[dof_local]

values[:] = x[:, 0]**2 + x[:, 1]**3

vec.set_local(values)
vec.apply('insert')

# Check
u0 = df.interpolate(df.Expression('x[0]*x[0]+x[1]*x[1]*x[1]', degree=2), V)
u0.vector().axpy(-1, vec)

error = u0.vector().norm('linf')
if df.MPI.rank(df.MPI.comm_world) == 0: print(V.dim(), error)