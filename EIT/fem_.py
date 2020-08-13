# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes
""" 2D/3D FEM routines """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

from collections import namedtuple
import numpy as np
import numpy.linalg as la
from scipy import sparse
from pyeit.eit.fem import *
from pyeit.eit.utils import eit_scan_lines


class Forward(Forward):
    """ FEM forward computing code """

    def __init__(self, mesh, el_pos):
        """
        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes

        Parameters
        ----------
        mesh: dict
            mesh structure, {'node', 'element', 'perm'}
        el_pos: NDArray
            numbering of electrodes positions
            
        Note
        ----
        the nodes are continuous numbered, the numbering of an element is
        CCW (counter-clock-wise).
        """
        super().__init__(mesh, el_pos)

    def solve_eit(self, ex_mat=None, step=1, perm=None, parser=None, skip_jac=False):
        """
        EIT simulation, generate perturbation matrix and forward v

        Parameters
        ----------
        ex_mat: NDArray
            numLines x n_el array, stimulation matrix
        step: int
            the configuration of measurement electrodes (default: adjacent)
        perm: NDArray
            Mx1 array, initial x0. must be the same size with self.tri_perm
        parser: str
            if parser is 'fmmu', within each stimulation pattern, diff_pairs
            or boundary measurements are re-indexed and started
            from the positive stimulus electrode
            if parser is 'std', subtract_row start from the 1st electrode

        Returns
        -------
        jac: NDArray
            number of measures x n_E complex array, the Jacobian
        v: NDArray
            number of measures x 1 array, simulated boundary measures
        b_matrix: NDArray
            back-projection mappings (smear matrix)
        """
        # initialize/extract the scan lines (default: apposition)
        if ex_mat is None:
            ex_mat = eit_scan_lines(16, 8)

        # initialize the permittivity on element
        if perm is None:
            perm0 = self.tri_perm
        elif np.isscalar(perm):
            perm0 = np.ones(self.n_tri, dtype=np.float)
        else:
            assert perm.shape == (self.n_tri,)
            perm0 = perm

        # calculate f and Jacobian iteratively over all stimulation lines
        jac, v, b_matrix = [], [], []
        n_lines = ex_mat.shape[0]

        for i in range(n_lines):
            # FEM solver of one stimulation pattern, a row in ex_mat
            ex_line = ex_mat[i]
            f, jac_i = self.solve(ex_line, perm0, skip_jac)
            f_el = f[self.el_pos]

            # boundary measurements, subtract_row-voltages on electrodes
            diff_op = voltage_meter(ex_line, n_el=self.ne, step=step, parser=parser)
            v_diff = subtract_row(f_el, diff_op)
            if not skip_jac: jac_diff = subtract_row(jac_i, diff_op)

            # build bp projection matrix
            # 1. we can either smear at the center of elements, using
            #    >> fe = np.mean(f[self.tri], axis=1)
            # 2. or, simply smear at the nodes using f
            b = smear(f, f_el, diff_op)

            # append
            v.append(v_diff)
            if not skip_jac: jac.append(jac_diff)
            b_matrix.append(b)

        # update output, now you can call p.jac, p.v, p.b_matrix
        pde_result = namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
        p = pde_result(jac=np.vstack(jac) if jac else jac,
                       v=np.hstack(v),
                       b_matrix=np.vstack(b_matrix))
        return p

    def solve(self, ex_line, perm, skip_jac=False):
        """
        with one pos (A), neg(B) driven pairs, calculate and
        compute the potential distribution (complex-valued)
        
        TODO: the calculation of Jacobian can be skipped.
        TODO: handle CEM (complete electrode model)

        Parameters
        ----------
        ex_line: NDArray
            stimulation (scan) patterns/lines
        perm: NDArray
            permittivity on elements (initial)

        Returns
        -------
        f: NDArray
            potential on nodes
        J: NDArray
            Jacobian
        """
        # 1. calculate local stiffness matrix (on each element)
        ke = calculate_ke(self.pts, self.tri)

        # 2. assemble to global K
        kg = assemble_sparse(ke, self.tri, perm, self.n_pts, ref=self.ref)

        # 3. calculate electrode impedance matrix R = K^{-1}
        r_matrix = la.inv(kg)
        r_el = r_matrix[self.el_pos]

        # 4. solving nodes potential using boundary conditions
        b = super()._natural_boundary(ex_line)
        f = np.dot(r_matrix, b).ravel()

        # 5. build Jacobian matrix column wise (element wise)
        #    Je = Re*Ke*Ve = (nex3) * (3x3) * (3x1)
        if skip_jac:
            jac = None
        else:
            jac = np.zeros((self.ne, self.n_tri), dtype=perm.dtype)
            for (i, e) in enumerate(self.tri):
                jac[:, i] = np.dot(np.dot(r_el[:, e], ke[i]), f[e])

        return f, jac
