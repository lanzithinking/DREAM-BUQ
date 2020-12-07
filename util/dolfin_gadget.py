#!/usr/bin/env python
"""
Some handy functions to facilitate usage of dolfin
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Modified Sept 2019 @ ASU
"""

import dolfin as df
import numpy as np

def create_PETScMatrix(shape,mpi_comm=None,rows=None,cols=None,values=None):
    """
    Create and set up PETScMatrix of arbitrary size using petsc4py.
    """
    if df.has_petsc4py():
        from petsc4py import PETSc
    else:
        print('Dolfin is not compiled with petsc4py! Cannot create PETScMatrix of arbitrary size.')
        exit()
    if mpi_comm is None:
        mpi_comm = df.mpi_comm_world()
    mat = PETSc.Mat()
    mat.create(mpi_comm)
    mat.setSizes(shape)
    mat.setType('aij')
    mat.setUp()
    mat.setValues(rows,cols,values)
    mat.assemble()
    return mat

def get_dof_coords(V):
    """
    Get the coordinates of dofs.
    """
    try:
        dof_coords = V.tabulate_dof_coordinates() # post v1.6.0
    except AttributeError:
        print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
        dof_coords = V.dofmap().tabulate_all_coordinates(V.mesh())
    dof_coords.resize((V.dim(), V.mesh().geometry().dim()),refcheck=False)
    return dof_coords

def check_in_dof(points,V,tol=2*df.DOLFIN_EPS):
    """
    Check whether points are nodes where dofs are defined and output those dofs
    """
    # V should NOT be mixed function space! Unless you know what you are doing...
    if V.num_sub_spaces()>1:
        print('Warning: Multiple dofs associated with each point, unreliable outputs!')
    # obtain coordinates of dofs
    dof_coords=get_dof_coords(V)
    # check whether those points are close to nodes where dofs are defined
    pdist_pts2dofs = np.einsum('ijk->ij',(points[:,None,:]-dof_coords[None,:,:])**2)
    idx_in_dof = np.argmin(pdist_pts2dofs,axis=1)
    rel_idx_in = np.where(np.einsum('ii->i',pdist_pts2dofs[:,idx_in_dof])<tol**2)[0] # index relative to points
    idx_in_dof = idx_in_dof[rel_idx_in]
    loc_in_dof = points[rel_idx_in,]
    return idx_in_dof,loc_in_dof,rel_idx_in

def check_in_mesh(points,mesh,tol=2*df.DOLFIN_EPS):
    """
    Check whether points are on a given mesh and output those mesh indices
    """
    # obtain mesh coordinates
    msh_coords=mesh.coordinates()
    # check whether those points are close to vertices of the mesh
    pdist_pts2mshs = np.einsum('ijk->ij',(points[:,None,:]-msh_coords[None,:,:])**2)
    idx_in_msh = np.argmin(pdist_pts2mshs,axis=1)
    rel_idx_in = np.where(np.einsum('ii->i',pdist_pts2mshs[:,idx_in_msh])<tol**2)[0] # index relative to points
    idx_in_msh = idx_in_msh[rel_idx_in]
    loc_in_msh = points[rel_idx_in,]
    return idx_in_msh,loc_in_msh,rel_idx_in

def vec2fun(vec,V):
    """
    Convert a vector to a dolfin function such that the function has the vector as coefficients.
    """
    f = df.Function(V)
#     f.vector()[:] = np.array(vec)
#     f.vector().set_local(vec[df.vertex_to_dof_map(V)]) # not working for CG 2
#     f.vector().set_local(vec[V.dofmap().dofs()])
    dofmap = V.dofmap()
    dof_first, dof_last = dofmap.ownership_range()
    unowned = dofmap.local_to_global_unowned()
    dofs = filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned, range(dof_last-dof_first))
#     dof_local=np.array(list(dofs))
#     vec_local=vec.get_local()
#     import pydevd; pydevd.settrace()
    f.vector().set_local(vec[list(dofs)])
#     f.vector().set_local(vec.get_local())
#     f.vector().apply('insert')
#     f.vector().zero()
#     f.vector().axpy(1.,vec)
    return f
 
def mat2fun(mat,V):
    """
    Convert a matrix (multiple vectors) to a dolfin mixed function such that each column corresponds to a component function.
    """
    k = mat.shape[1]
    # mixed functions to store functions
    M=df.MixedFunctionSpace([V]*k)
    # Extract subfunction dofs
    dofs = [M.sub(i).dofmap().dofs() for i in range(k)]
    f=df.Function(M)
    for i,dof_i in enumerate(dofs):
        f.vector()[dof_i]=mat[:,i]
    return f,dofs

def fun2img(f):
    """
    Obtain the pixel matrix of an image from a dolfin function
    """
    mesh = f.function_space().mesh()
    gdim = mesh.geometry().dim()
    # DG0 cellwise function
    if f.vector().size() == mesh.num_cells():
        C = f.vector().get_local()
    # Scalar function, interpolated to vertices
    elif f.value_rank() == 0:
        C = f.compute_vertex_values(mesh)
    # Vector function, interpolated to vertices
    elif f.value_rank() == 1:
        w0 = f.compute_vertex_values(mesh)
        nv = mesh.num_vertices()
        if len(w0) != gdim * nv:
            raise AttributeError('Vector length must match geometric dimension.')
#         X = mesh.coordinates()
#         X = [X[:, i] for i in range(gdim)]
        U = [w0[i * nv: (i + 1) * nv] for i in range(gdim)]
        # Compute magnitude
        C = U[0]**2
        for i in range(1, gdim):
            C += U[i]**2
        C = np.sqrt(C)
    else:
        raise AttributeError('Wrong function input!')
    imsz = np.floor(C.size**(1./gdim)).astype('int')
    im_shape=(-1,)+(imsz,)*(gdim-1)
    return C.reshape(im_shape)
    
def img2fun(im,V):
    """
    Convert an image to a dolfin function in given space
    """
    f = df.Function(V)
    mesh = V.mesh()
    # DG0 cellwise function
    if f.vector().size() == mesh.num_cells():
        f.vector().set_local(im.flatten()) # pay attention to the order ('C' or 'F')
    # Scalar function, interpolated to vertices
    elif f.value_rank() == 0:
        if V.ufl_element().degree()==1:
            d2v = df.dof_to_vertex_map(V)
            f.vector().set_local(im.flatten()[d2v])
        elif V.ufl_element().degree()>1:
            V_deg1 = df.FunctionSpace(mesh, V.ufl_element().family(), 1)
#             d2v = df.dof_to_vertex_map(V_deg1)
#             im_f = df.Function(V_deg1)
#             im_f.vector().set_local(im.flatten()[d2v])
            im_f = img2fun(im, V_deg1)
            f.interpolate(im_f)
    else:
        raise AttributeError('Not supported!')
    return f

# functions to convert vectors between P1 and Pn
def vinP1(v, V):
    """project v from Pn to P1 space"""
    if len(v)==V.dim():
        vec = v
    else:
        mesh = V.mesh()
        V_P1 = df.FunctionSpace(mesh, V.ufl_element().family(), 1)
        d2v = df.dof_to_vertex_map(V_P1)
        f = df.Function(V)
        f.vector().set_local(v)
    #     vec = df.Function(V_P1).vector()
        vec = df.Vector(mesh.mpi_comm(),mesh.num_vertices())
        vec.set_local(f.compute_vertex_values(mesh)[d2v])
    return vec

def vinPn(v, V):
    """project v from P1 to Pn space"""
    if len(v)==V.dim():
        vec = v
    else:
        V_P1 = df.FunctionSpace(V.mesh(), V.ufl_element().family(), 1)
        f_P1 = df.Function(V_P1)
        f_P1.vector().set_local(v)
        f = df.Function(V)
        f.interpolate(f_P1)
        vec = f.vector()
    return vec