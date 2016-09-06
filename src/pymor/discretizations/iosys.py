# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.lyapunov import solve_lyap
from pymor.algorithms.riccati import solve_ricc
from pymor.algorithms.to_matrix import to_matrix
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.block import BlockOperator, BlockDiagonalOperator
from pymor.operators.constructions import (Concatenation, IdentityOperator, LincombOperator, ZeroOperator)
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.tools.frozendict import FrozenDict
from pymor.vectorarrays.numpy import NumpyVectorArray


class InputOutputSystem(DiscretizationInterface):
    """Base class for input-output systems.

    Attributes
    ----------
    m
        The number of inputs.
    p
        The number of outputs.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    """
    m = p = 0
    cont_time = True

    def _solve(self, mu=None):
        raise NotImplementedError

    def eval_tf(self, s, mu=None):
        """Evaluate the transfer function of the system."""
        raise NotImplementedError

    def eval_dtf(self, s, mu=None):
        """Evaluate the derivative of the transfer function of the system."""
        raise NotImplementedError

    def bode(self, w):
        """Evaluate the transfer function on the imaginary axis.

        Parameters
        ----------
        w
            Frequencies at which to compute the transfer function.

        Returns
        -------
        tfw
            Transfer function values at frequencies in `w`,
            |NumPy array| of shape `(self.p, self.m, len(w))`.
        """
        if not self.cont_time:
            raise NotImplementedError

        self._w = w
        self._tfw = np.dstack([self.eval_tf(1j * wi) for wi in w])
        return self._tfw.copy()


class LTISystem(InputOutputSystem):
    r"""Class for linear time-invariant systems.

    This class describes input-state-output systems given by

    .. math::
        E x'(t) &= A x(t) + B u(t) \\
           y(t) &= C x(t) + D u(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + B u(k) \\
          y(k)     &= C x(k) + D u(k)

    if discrete-time, where :math:`A`, :math:`B`, :math:`C`, :math:`D`, and
    :math:`E` are linear operators.

    Parameters
    ----------
    A
        The |Operator| A.
    B
        The |Operator| B.
    C
        The |Operator| C.
    D
        The |Operator| D or `None` (then D is assumed to be zero).
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    ss_operators
        A dictonary for state-to-state |Operators| A and E.
    is_operators
        A dictonary for input-to-state |Operator| B.
    so_operators
        A dictonary for state-to-output |Operator| C.
    io_operators
        A dictonary for input-to-output |operator| D.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.

    Attributes
    ----------
    n
        The order of the system.
    m
        The number of inputs.
    p
        The number of outputs.
    A
        The |Operator| A. The same as `ss_operators['A']`.
    B
        The |Operator| B. The same as `is_operators['B']`.
    C
        The |Operator| C. The same as `so_operators['C']`.
    D
        The |Operator| D. The same as `io_operators['D']`.
    E
        The |Operator| E. The same as `ss_operators['E']`.
    """
    linear = True

    def __init__(self, A=None, B=None, C=None, D=None, E=None, ss_operators=None, is_operators=None,
                 so_operators=None, io_operators=None, cont_time=True):
        A = A or ss_operators['A']
        B = B or is_operators['B']
        C = C or so_operators['C']
        D = D or io_operators.get('D')
        E = E or ss_operators.get('E')
        assert isinstance(A, OperatorInterface) and A.linear
        assert A.source == A.range
        assert isinstance(B, OperatorInterface) and B.linear
        assert B.range == A.source
        assert isinstance(C, OperatorInterface) and C.linear
        assert C.source == A.range
        assert (D is None or
                isinstance(D, OperatorInterface) and D.linear and D.source == B.source and D.range == C.range)
        assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        self.n = A.source.dim
        self.m = B.source.dim
        self.p = C.range.dim
        self.A = A
        self.B = B
        self.C = C
        self.D = D if D is not None else ZeroOperator(B.source, C.range)
        self.E = E if E is not None else IdentityOperator(A.source)
        self.ss_operators = FrozenDict({'A': A, 'E': self.E})
        self.is_operators = FrozenDict({'B': B})
        self.so_operators = FrozenDict({'C': C})
        self.io_operators = FrozenDict({'D': self.D})
        self.cont_time = cont_time
        self._poles = None
        self._w = None
        self._tfw = None
        self._gramian = {}
        self._sv = {}
        self._U = {}
        self._V = {}
        self._H2_norm = None
        self._Hinf_norm = None
        self._fpeak = None
        self.build_parameter_type(inherits=(A, B, C, D, E))

    @classmethod
    def from_matrices(cls, A, B, C, D=None, E=None, cont_time=True):
        """Create |LTISystem| from matrices.

        Parameters
        ----------
        A
            The |NumPy array| or |SciPy spmatrix| A.
        B
            The |NumPy array| or |SciPy spmatrix| B.
        C
            The |NumPy array| or |SciPy spmatrix| C.
        D
            The |NumPy array| or |SciPy spmatrix| D or `None` (then D is assumed
            to be zero).
        E
            The |NumPy array| or |SciPy spmatrix| E or `None` (then E is assumed
            to be identity).
        cont_time
            `True` if the system is continuous-time, otherwise `False`.

        Returns
        -------
        lti
            The |LTISystem| with operators A, B, C, D, and E.
        """
        assert isinstance(A, (np.ndarray, sps.spmatrix))
        assert isinstance(B, (np.ndarray, sps.spmatrix))
        assert isinstance(C, (np.ndarray, sps.spmatrix))
        assert D is None or isinstance(D, (np.ndarray, sps.spmatrix))
        assert E is None or isinstance(E, (np.ndarray, sps.spmatrix))

        A = NumpyMatrixOperator(A)
        B = NumpyMatrixOperator(B)
        C = NumpyMatrixOperator(C)
        if D is not None:
            D = NumpyMatrixOperator(D)
        if E is not None:
            E = NumpyMatrixOperator(E)

        return cls(A, B, C, D, E, cont_time)

    @classmethod
    def from_files(cls, A_file, B_file, C_file, D_file=None, E_file=None, cont_time=True):
        """Create |LTISystem| from matrices stored in separate files.

        Parameters
        ----------
        A_file
            The name of the file (with extension) containing A.
        B_file
            The name of the file (with extension) containing B.
        C_file
            The name of the file (with extension) containing C.
        D_file
            `None` or the name of the file (with extension) containing D.
        E_file
            `None` or the name of the file (with extension) containing E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.

        Returns
        -------
        lti
            The |LTISystem| with operators A, B, C, D, and E.
        """
        from pymor.tools.io import load_matrix

        A = load_matrix(A_file)
        B = load_matrix(B_file)
        C = load_matrix(C_file)
        D = load_matrix(D_file) if D_file is not None else None
        E = load_matrix(E_file) if E_file is not None else None

        return cls.from_matrices(A, B, C, D, E, cont_time)

    @classmethod
    def from_mat_file(cls, file_name, cont_time=True):
        """Create |LTISystem| from matrices stored in a .mat file.

        Parameters
        ----------
        file_name
            The name of the .mat file (extension .mat does not need to be
            included) containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.

        Returns
        -------
        lti
            The |LTISystem| with operators A, B, C, D, and E.
        """
        import scipy.io as spio
        mat_dict = spio.loadmat(file_name)

        assert 'A' in mat_dict and 'B' in mat_dict and 'C' in mat_dict

        A = mat_dict['A']
        B = mat_dict['B']
        C = mat_dict['C']
        D = mat_dict['D'] if 'D' in mat_dict else None
        E = mat_dict['E'] if 'E' in mat_dict else None

        return cls.from_matrices(A, B, C, D, E, cont_time)

    @classmethod
    def from_abcde_files(cls, files_basename, cont_time=True):
        """Create |LTISystem| from matrices stored in a .[ABCDE] files.

        Parameters
        ----------
        files_basename
            The basename of files containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.

        Returns
        -------
        lti
            The |LTISystem| with operators A, B, C, D, and E.
        """
        from pymor.tools.io import load_matrix
        import os.path

        A = load_matrix(files_basename + '.A')
        B = load_matrix(files_basename + '.B')
        C = load_matrix(files_basename + '.C')
        D = load_matrix(files_basename + '.D') if os.path.isfile(files_basename + '.D') else None
        E = load_matrix(files_basename + '.E') if os.path.isfile(files_basename + '.E') else None

        return cls.from_matrices(A, B, C, D, E, cont_time)

    def __add__(self, other):
        """Add two |LTISystems|."""
        assert isinstance(other, LTISystem)
        assert self.B.source == other.B.source
        assert self.C.range == other.C.range
        assert self.cont_time == other.cont_time

        A = BlockDiagonalOperator((self.A, other.A))
        B = BlockOperator.vstack((self.B, other.B))
        C = BlockOperator.hstack((self.C, other.C))
        D = (self.D + other.D).assemble()
        E = BlockDiagonalOperator((self.E, other.E))

        return self.__class__(A, B, C, D, E, self.cont_time)

    def __neg__(self):
        """Negate |LTISystem|."""
        A = self.A
        B = self.B
        C = (self.C * (-1)).assemble()
        D = (self.D * (-1)).assemble()
        E = self.E

        return self.__class__(A, B, C, D, E, self.cont_time)

    def __sub__(self, other):
        """Subtract two |LTISystems|."""
        return self + (-other)

    def __mul__(self, other):
        """Multiply (cascade) two |LTISystems|."""
        assert self.B.source == other.C.range

        A = BlockOperator([[self.A, Concatenation(self.B, other.C)],
                           [None, other.A]])
        B = BlockOperator.vstack((Concatenation(self.B, other.D),
                                  other.B))
        C = BlockOperator.hstack((self.C, Concatenation(self.D,
                                  other.C)))
        D = Concatenation(self.D, other.D)
        E = BlockDiagonalOperator((self.E, other.E))

        return self.__class__(A, B, C, D, E, self.cont_time)

    def compute_poles(self):
        """Compute system poles."""
        if self._poles is None:
            A = to_matrix(self.A)
            E = None if isinstance(self.E, IdentityOperator) else to_matrix(self.E)
            self._poles = spla.eigvals(A, E)

    def eval_tf(self, s):
        """Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            C (s E - A)^{-1} B + D.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array|
            of shape `(self.p, self.m)`.
        """
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        sEmA = LincombOperator((E, A), (s, -1))
        if self.m <= self.p:
            I_m = B.source.from_data(sp.eye(self.m))
            tfs = C.apply(sEmA.apply_inverse(B.apply(I_m))).data.T
        else:
            I_p = C.range.from_data(sp.eye(self.p))
            tfs = B.apply_adjoint(sEmA.apply_inverse_adjoint(C.apply_adjoint(I_p))).data.conj()
        if not isinstance(D, ZeroOperator):
            if self.m <= self.p:
                tfs += D.apply(I_m).data.T
            else:
                tfs += D.apply_adjoint(I_p).data
        return tfs

    def eval_dtf(self, s):
        """Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            -C (s E - A)^{-1} E (s E - A)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        A = self.A
        B = self.B
        C = self.C
        E = self.E

        sEmA = LincombOperator((E, A), (s, -1))
        if self.m <= self.p:
            I_m = B.source.from_data(sp.eye(self.m))
            dtfs = -C.apply(sEmA.apply_inverse(E.apply(sEmA.apply_inverse(B.apply(I_m))))).data.T
        else:
            I_p = C.range.from_data(sp.eye(self.p))
            dtfs = B.apply_adjoint(sEmA.apply_inverse_adjoint(E.apply_adjoint(sEmA.apply_inverse_adjoint(
                C.apply_adjoint(I_p))))).data.conj()
        return dtfs

    @classmethod
    def mag_plot(cls, sys_list, plot_style_list=None, w=None, ord=None, dB=False, Hz=False):
        """Draw the magnitude Bode plot.

        Parameters
        ----------
        sys_list
            A single |LTISystem| or a list of |LTISystems|.
        plot_style_list
            A string or a list of strings of the same length as `sys_list`.

            If `None`, matplotlib defaults are used.
        w
            Frequencies at which to compute the transfer function.

            If `None`, use `self._w`.
        ord
            The order of the norm used to compute the magnitude (the default is
            the Frobenius norm).
        dB
            Should the magnitude be in dB on the plot.
        Hz
            Should the frequency be in Hz on the plot.

        Returns
        -------
        fig
            Matplotlib figure.
        ax
            Matplotlib axes.
        """
        assert isinstance(sys_list, LTISystem) or all(isinstance(sys, LTISystem) for sys in sys_list)
        if isinstance(sys_list, LTISystem):
            sys_list = (sys_list,)

        assert (plot_style_list is None or isinstance(plot_style_list, str) or
                all(isinstance(plot_style, str) for plot_style in plot_style_list))
        if isinstance(plot_style_list, str):
            plot_style_list = (plot_style_list,)

        assert w is not None or all(sys._w is not None for sys in sys_list)

        if w is not None:
            for sys in sys_list:
                sys.bode(w)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i, sys in enumerate(sys_list):
            freq = sys._w / (2 * np.pi) if Hz else sys._w
            mag = spla.norm(sys._tfw, ord=ord, axis=(0, 1))
            style = '' if plot_style_list is None else plot_style_list[i]
            if dB:
                mag = 20 * np.log2(mag)
                ax.semilogx(freq, mag, style)
            else:
                ax.loglog(freq, mag, style)
        ax.set_title('Magnitude Bode Plot')
        if Hz:
            ax.set_xlabel('Frequency (Hz)')
        else:
            ax.set_xlabel('Frequency (rad/s)')
        if dB:
            ax.set_ylabel('Magnitude (dB)')
        else:
            ax.set_ylabel('Magnitude')
        plt.show()
        return fig, ax

    def compute_gramian(self, typ, subtyp, me_solver=None, tol=None):
        """Compute a Gramian.

        Parameters
        ----------
        typ
            The type of the Gramian:

            - `'lyap'`: Lyapunov Gramian,
            - `'lqg'`: LQG Gramian,
            - `('br', gamma)`: Bounded Real Gramian with parameter gamma.
        subtyp
            The subtype of the Gramian:

            - `'cf'`: controllability Gramian factor,
            - `'of'`: observability Gramian factor.
        me_solver
            The matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        tol
            The tolerance parameter for the low-rank matrix equation solver.

            If `None`, then the default tolerance is used. Otherwise, it should
            be a positive float and the Gramian factor is recomputed (if it was
            already computed).
        """
        assert isinstance(typ, (str, tuple))
        assert isinstance(subtyp, str)

        if not self.cont_time:
            raise NotImplementedError

        if typ not in self._gramian or subtyp not in self._gramian[typ] or tol is not None:
            A = self.A
            B = self.B
            C = self.C
            E = self.E if not isinstance(self.E, IdentityOperator) else None
            if typ == 'lyap':
                if subtyp == 'cf':
                    self._gramian[typ][subtyp] = solve_lyap(A, E, B, trans=False, me_solver=me_solver, tol=tol)
                elif subtyp == 'of':
                    self._gramian[typ][subtyp] = solve_lyap(A, E, C, trans=True, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for 'lyap' type.")
            elif typ == 'lqg':
                if subtyp == 'cf':
                    self._gramian[typ][subtyp] = solve_ricc(A, E=E, B=B, C=C, trans=True, me_solver=me_solver, tol=tol)
                elif subtyp == 'of':
                    self._gramian[typ][subtyp] = solve_ricc(A, E=E, B=B, C=C, trans=False, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for 'lqg' type.")
            elif isinstance(typ, tuple) and typ[0] == 'br':
                assert isinstance(typ[1], float)
                assert typ[1] > 0
                if subtyp == 'cf':
                    self._gramian[typ][subtyp] = solve_ricc(A, E=E, B=B / np.sqrt(typ[1]), C=C / np.sqrt(typ[1]),
                                                            R=-IdentityOperator(C.range),
                                                            trans=True, me_solver=me_solver, tol=tol)
                elif subtyp == 'of':
                    self._gramian[typ][subtyp] = solve_ricc(A, E=E, B=B / np.sqrt(typ[1]), C=C / np.sqrt(typ[1]),
                                                            R=-IdentityOperator(B.source),
                                                            trans=False, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for ('br', gamma) type.")
            else:
                raise NotImplementedError("Only 'lyap', 'lqg', and ('br', gamma) types are available.")

    def compute_sv_U_V(self, typ, me_solver=None):
        """Compute singular values and vectors.

        Parameters
        ----------
        typ
            The type of the Gramian (see
            :func:`~pymor.discretizations.iosys.LTISystem.compute_gramian`).
        me_solver
            The matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        """
        assert isinstance(typ, tuple)

        if typ not in self._sv or typ not in self._U or typ not in self._V:
            self.compute_gramian(typ, 'cf', me_solver=me_solver)
            self.compute_gramian(typ, 'of', me_solver=me_solver)

            U, self._sv[typ], Vh = spla.svd(self.E.apply2(self._gramian[typ]['of'], self._gramian[typ]['cf']))

            self._U[typ] = NumpyVectorArray(U.T)
            self._V[typ] = NumpyVectorArray(Vh)

    def norm(self, name='H2'):
        """Compute a norm of the |LTISystem|.

        Parameters
        ----------
        name
            The name of the norm (`'H2'`, `'Hinf'`, `'Hankel'`).
        """
        if name == 'H2':
            if self._H2_norm is not None:
                return self._H2_norm

            B, C = self.B, self.C

            if 'lyap' in self._gramian and 'cf' in self._gramian['lyap']:
                self._H2_norm = np.sqrt(C.apply(self._gramian['lyap']['cf']).l2_norm2().sum())
            elif 'lyap' in self._gramian and 'of' in self._gramian['lyap']:
                self._H2_norm = np.sqrt(B.apply_adjoint(self._gramian['lyap']['of']).l2_norm2().sum())
            elif self.m <= self.p:
                self.compute_gramian('lyap', 'cf')
                self._H2_norm = np.sqrt(C.apply(self._gramian['lyap']['cf']).l2_norm2().sum())
            else:
                self.compute_gramian('lyap', 'of')
                self._H2_norm = np.sqrt(B.apply_adjoint(self._gramian['lyap']['of']).l2_norm2().sum())

            return self._H2_norm
        elif name == 'Hinf':
            if self._Hinf_norm is not None:
                return self._Hinf_norm

            from slycot import ab13dd
            dico = 'C' if self.cont_time else 'D'
            jobe = 'I' if isinstance(self.E, IdentityOperator) else 'G'
            equil = 'S'
            jobd = 'Z' if isinstance(self.D, ZeroOperator) else 'D'
            A, B, C, D, E = map(to_matrix, (self.A, self.B, self.C,
                                            self.D, self.E))
            self._Hinf_norm, self._fpeak = ab13dd(dico, jobe, equil, jobd, self.n, self.m, self.p, A, E, B, C, D)

            return self._Hinf_norm
        elif name == 'Hankel':
            self.compute_sv_U_V('lyap')
            return self._sv['lyap'][0]
        else:
            raise NotImplementedError('Only H2, Hinf, and Hankel norms are implemented.')


class TF(InputOutputSystem):
    """Class for input-output systems represented by a transfer function.

    This class describes input-output systems given by a transfer function
    :math:`H(s)`.

    Parameters
    ----------
    m
        The number of inputs.
    p
        The number of outputs.
    H
        The transfer function defined at least on the open right complex
        half-plane.

        `H(s)` is a |NumPy array| of shape `(p, m)`.
    dH
        The complex derivative of `H`.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    """
    linear = True

    def __init__(self, m, p, H, dH, cont_time=True):
        assert cont_time in (True, False)

        self.m = m
        self.p = p
        self.tf = H
        self.dtf = dH
        self.cont_time = cont_time
        self._w = None
        self._tfw = None

    def eval_tf(self, s):
        return self.tf(s)

    def eval_dtf(self, s):
        return self.dtf(s)


class SecondOrderSystem(InputOutputSystem):
    r"""Class for linear second order systems.

    This class describes input-output systems given by

    .. math::
        M x''(t) + D x'(t) + K x(t) &= B u(t) \\
                               y(t) &= C_p x(t) + C_v x'(t)

    if continuous-time, or

    .. math::
        M x(k + 2) + D x(k + 1) + K x(k) &= B u(k) \\
                                    y(k) &= C_p x(k) + C_v x(k + 1)

    if discrete-time, where :math:`M`, :math:`D`, :math:`K`, :math:`B`,
    :math:`C_p`, and :math:`C_v` are linear operators.

    Parameters
    ----------
    M
        The |Operator| M or `None` (then M is assumed to be identity).
    D
        The |Operator| D or `None` (then D is assumed to be zero).
    K
        The |Operator| K.
    B
        The |Operator| B.
    Cp
        The |Operator| Cp.
    Cv
        The |Operator| Cv.
    ss_operators
        A dictonary for state-to-state |Operators| M, D, and K.
    is_operators
        A dictonary for input-to-state |Operator| B.
    so_operators
        A dictonary for state-to-output |Operators| Cp and Cv.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.

    Attributes
    ----------
    n
        The order of the system (equal to M.source.dim).
    m
        The number of inputs.
    p
        The number of outputs.
    M
        The |Operator| M. The same as `ss_operators['M']`.
    D
        The |Operator| D. The same as `ss_operators['D']`.
    K
        The |Operator| K. The same as `ss_operators['K']`.
    B
        The |Operator| B. The same as `is_operators['B']`.
    Cp
        The |Operator| Cp. The same as `so_operators['Cp']`.
    Cv
        The |Operator| Cv. The same as `so_operators['Cv']`.
    """
    linear = True

    def __init__(self, M=None, D=None, K=None, B=None, Cp=None, Cv=None, ss_operators=None, is_operators=None,
                 so_operators=None, io_operators=None, cont_time=True):
        M = M or ss_operators.get('M')
        D = D or ss_operators.get('D')
        K = K or ss_operators['D']
        B = B or is_operators['B']
        Cp = Cp or so_operators['Cp']
        Cv = Cv or so_operators['Cv']
        assert isinstance(K, OperatorInterface) and K.linear
        assert K.source == K.range
        assert isinstance(B, OperatorInterface) and B.linear
        assert B.range == K.source
        assert isinstance(Cp, OperatorInterface) and Cp.linear
        assert Cp.source == K.range
        assert isinstance(Cv, OperatorInterface) and Cv.linear
        assert Cv.source == K.range
        assert Cp.range == Cv.range
        assert M is None or isinstance(M, OperatorInterface) and M.linear and M.source == M.range == K.source
        assert D is None or isinstance(D, OperatorInterface) and D.linear and D.source == D.range == K.source
        assert cont_time in (True, False)

        self.n = K.source.dim
        self.m = B.source.dim
        self.p = Cp.range.dim
        self.M = M if M is not None else IdentityOperator(K.source)
        self.D = D if D is not None else ZeroOperator(K.source, K.range)
        self.K = K
        self.B = B
        self.Cp = Cp
        self.Cv = Cv
        self.ss_operators = FrozenDict({'M': self.M, 'D': self.D, 'K': K})
        self.is_operators = FrozenDict({'B': B})
        self.so_operators = FrozenDict({'Cp': Cp, 'Cv': Cv})
        self.io_operators = FrozenDict({})
        self.cont_time = cont_time
        self._poles = None
        self._w = None
        self._tfw = None
        self._gramian = {}
        self._sv = {}
        self._U = {}
        self._V = {}
        self._H2_norm = None
        self._Hinf_norm = None
        self._fpeak = None
        self.build_parameter_type(inherits=(M, D, K, B, Cp, Cv))

    def eval_tf(self, s):
        """Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            (C_p + s C_v) (s^2 M + s D + K)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array|
            of shape `(self.p, self.m)`.
        """
        M = self.M
        D = self.D
        K = self.K
        B = self.B
        Cp = self.Cp
        Cv = self.Cv

        s2MpsDpK = LincombOperator((M, D, K), (s ** 2, s, 1))
        if self.m <= self.p:
            I_m = B.source.from_data(sp.eye(self.m))
            CppsCv = LincombOperator((Cp, Cv), (1, s))
            tfs = CppsCv.apply(s2MpsDpK.apply_inverse(B.apply(I_m))).data.T
        else:
            I_p = Cp.range.from_data(sp.eye(self.p))
            tfs = B.apply_adjoint(s2MpsDpK.apply_inverse_adjoint(Cp.apply_adjoint(I_p) +
                                                                 Cv.apply_adjoint(I_p) * s.conj())).data.conj()
        return tfs

    def eval_dtf(self, s):
        """Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            s C_v (s^2 M + s D + K)^{-1} B
            - (C_p + s C_v) (s^2 M + s D + K)^{-1} (2 s M + D)
                (s^2 M + s D + K)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        M = self.M
        D = self.D
        K = self.K
        B = self.B
        Cp = self.Cp
        Cv = self.Cv

        s2MpsDpK = LincombOperator((M, D, K), (s ** 2, s, 1))
        sM2pD = LincombOperator((M, D), (2 * s, 1))
        if self.m <= self.p:
            I_m = B.source.from_data(sp.eye(self.m))
            dtfs = Cv.apply(s2MpsDpK.apply_inverse(B.apply(I_m))).data.T * s
            CppsCv = LincombOperator((Cp, Cv), (1, s))
            dtfs -= CppsCv.apply(s2MpsDpK.apply_inverse(sM2pD.apply(s2MpsDpK.apply_inverse(B.apply(I_m))))).data.T
        else:
            I_p = Cp.range.from_data(sp.eye(self.p))
            dtfs = B.apply_adjoint(s2MpsDpK.apply_inverse_adjoint(Cv.apply_adjoint(I_m))).data.conj() * s
            dtfs -= B.apply_adjoint(s2MpsDpK.apply_inverse_adjoint(sM2pD.apply_adjoint(s2MpsDpK.apply_inverse_adjoint(
                Cp.apply_adjoint(I_p) + Cv.apply_adjoint(I_p) * s.conj())))).data.conj()
        return dtfs


class LinearDelaySystem(InputOutputSystem):
    r"""Class for linear delay systems.

    This class describes input-output systems given by

    .. math::
        E x'(t) &= A x(t) + \sum_{i = 1}^q{A_i x(t - \tau_i)} + B u(t) \\
           y(t) &= C x(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + \sum_{i = 1}^q{A_i x(k - \tau_i)} + B u(k) \\
              y(k) &= C x(k)

    if discrete-time, where :math:`E`, :math:`A`, :math:`A_i`, :math:`B`,
    and :math:`C` are linear operators.

    Parameters
    ----------
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    A
        The |Operator| A.
    Ad
        The tuple of |Operators| A_i.
    tau
        The tuple of delay times (positive floats or ints).
    B
        The |Operator| B.
    C
        The |Operator| C.
    ss_operators
        A dictonary for state-to-state |Operators| E, A, and Ad.
    is_operators
        A dictonary for input-to-state |Operator| B.
    so_operators
        A dictonary for state-to-output |Operator| C.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.

    Attributes
    ----------
    n
        The order of the system (equal to A.source.dim).
    m
        The number of inputs.
    p
        The number of outputs.
    q
        The number of delay terms.
    E
        The |Operator| E. The same as `ss_operators['E']`.
    A
        The |Operator| A. The same as `ss_operators['A']`.
    Ad
        The tuple of |Operators| A_i. The same as `ss_operators['Ad']`.
    B
        The |Operator| B. The same as `is_operators['B']`.
    C
        The |Operator| C. The same as `so_operators['C']`.
    """
    linear = True

    def __init__(self, E=None, A=None, Ad=None, tau=None, B=None, C=None, ss_operators=None, is_operators=None,
                 so_operators=None, io_operators=None, cont_time=True):
        E = E or ss_operators.get('E')
        A = A or ss_operators['A']
        Ad = Ad or ss_operators['Ad']
        B = B or is_operators['B']
        C = C or so_operators['C']
        assert isinstance(A, OperatorInterface) and A.linear
        assert A.source == A.range
        assert isinstance(Ad, tuple)
        assert len(Ad) > 0
        assert all(isinstance(Ai, OperatorInterface) and Ai.linear for Ai in Ad)
        assert all(Ai.source == Ai.range == A.source for Ai in Ad)
        assert isinstance(tau, tuple)
        assert len(tau) == len(Ad)
        assert all(taui > 0 for taui in tau)
        assert isinstance(B, OperatorInterface) and B.linear
        assert B.range == A.source
        assert isinstance(C, OperatorInterface) and C.linear
        assert C.source == A.range
        assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        self.n = A.source.dim
        self.m = B.source.dim
        self.p = C.range.dim
        self.q = len(Ad)
        self.E = E if E is not None else IdentityOperator(A.source)
        self.A = A
        self.Ad = Ad
        self.B = B
        self.C = C
        self.ss_operators = FrozenDict({'E': self.E, 'A': A, 'Ad': Ad})
        self.is_operators = FrozenDict({'B': B})
        self.so_operators = FrozenDict({'C': C})
        self.io_operators = FrozenDict({})
        self.cont_time = cont_time
        self._poles = None
        self._w = None
        self._tfw = None
        self._gramian = {}
        self._sv = {}
        self._U = {}
        self._V = {}
        self._H2_norm = None
        self._Hinf_norm = None
        self._fpeak = None
        self.build_parameter_type(inherits=(E, A, Ad, B, C))

    def eval_tf(self, s):
        r"""Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            C \left(s E - A - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array|
            of shape `(self.p, self.m)`.
        """
        E = self.E
        A = self.A
        Ad = self.Ad
        B = self.B
        C = self.C

        middle = LincombOperator((E, A) + Ad, (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau))
        if self.m <= self.p:
            I_m = B.source.from_data(sp.eye(self.m))
            tfs = C.apply(middle.apply_inverse(B.apply(I_m))).data.T
        else:
            I_p = C.range.from_data(sp.eye(self.p))
            tfs = B.apply_adjoint(middle.apply_inverse_adjoint(C.apply_adjoint(I_p))).data.conj()
        return tfs

    def eval_dtf(self, s):
        r"""Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            -C \left(s E - A - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1}
                \left(E + \sum_{i = 1}^q{\tau_i e^{-\tau_i s} A_i}\right)
                \left(s E - A - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        E = self.E
        A = self.A
        Ad = self.Ad
        B = self.B
        C = self.C

        left_and_right = LincombOperator((E, A) + Ad, (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau))
        middle = LincombOperator((E,) + Ad, (s,) + tuple(taui * np.exp(-taui * s) for taui in self.tau))
        if self.m <= self.p:
            I_m = B.source.from_data(sp.eye(self.m))
            dtfs = C.apply(left_and_right.apply_inverse(middle.apply(left_and_right.apply_inverse(
                B.apply(I_m))))).data.T
        else:
            I_p = C.range.from_data(sp.eye(self.p))
            dtfs = B.apply_adjoint(left_and_right.apply_inverse_adjoint(middle.apply_adjoint(
                left_and_right.apply_inverse_adjointi(C.apply_adjoint(I_p))))).data.conj()
        return dtfs


class LinearStochasticSystem(InputOutputSystem):
    r"""Class for linear stochastic systems.

    This class describes input-output systems given by

    .. math::
        E \mathrm{d}x(t) &= A x(t) \mathrm{d}t
                            + \sum_{i = 1}^q{A_i x(t) \mathrm{d}\omega_i(t)}
                            + B u(t) \mathrm{d}t \\
                    y(t) &= C x(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + \sum_{i = 1}^q{A_i x(k) \omega_i(k)} + B u(k) \\
              y(k) &= C x(k)

    if discrete-time, where :math:`E`, :math:`A`, :math:`A_i`, :math:`B`,
    and :math:`C` are linear operators and :math:`\omega_i` are stochastic
    processes.

    Parameters
    ----------
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    A
        The |Operator| A.
    As
        The tuple of |Operators| A_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    ss_operators
        A dictonary for state-to-state |Operators| E, A, and As.
    is_operators
        A dictonary for input-to-state |Operator| B.
    so_operators
        A dictonary for state-to-output |Operator| C.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.

    Attributes
    ----------
    n
        The order of the system (equal to A.source.dim).
    m
        The number of inputs.
    p
        The number of outputs.
    q
        The number of stochastic processes.
    E
        The |Operator| E. The same as `ss_operators['E']`.
    A
        The |Operator| A. The same as `ss_operators['A']`.
    As
        The tuple of |Operators| A_i. The same as `ss_operators['As']`.
    B
        The |Operator| B. The same as `is_operators['B']`.
    C
        The |Operator| C. The same as `so_operators['C']`.
    """
    linear = True

    def __init__(self, E=None, A=None, As=None, tau=None, B=None, C=None, ss_operators=None, is_operators=None,
                 so_operators=None, io_operators=None, cont_time=True):
        E = E or ss_operators.get('E')
        A = A or ss_operators['A']
        As = As or ss_operators['As']
        B = B or is_operators['B']
        C = C or so_operators['C']
        assert isinstance(A, OperatorInterface) and A.linear
        assert A.source == A.range
        assert isinstance(As, tuple)
        assert len(As) > 0
        assert all(isinstance(Ai, OperatorInterface) and Ai.linear for Ai in As)
        assert all(Ai.source == Ai.range == A.source for Ai in As)
        assert isinstance(tau, tuple)
        assert len(tau) == len(As)
        assert all(taui > 0 for taui in tau)
        assert isinstance(B, OperatorInterface) and B.linear
        assert B.range == A.source
        assert isinstance(C, OperatorInterface) and C.linear
        assert C.source == A.range
        assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        self.n = A.source.dim
        self.m = B.source.dim
        self.p = C.range.dim
        self.q = len(As)
        self.E = E if E is not None else IdentityOperator(A.source)
        self.A = A
        self.As = As
        self.B = B
        self.C = C
        self.ss_operators = FrozenDict({'E': self.E, 'A': A, 'As': As})
        self.is_operators = FrozenDict({'B': B})
        self.so_operators = FrozenDict({'C': C})
        self.io_operators = FrozenDict({})
        self.cont_time = cont_time
        self._poles = None
        self._w = None
        self._tfw = None
        self._gramian = {}
        self._sv = {}
        self._U = {}
        self._V = {}
        self._H2_norm = None
        self._Hinf_norm = None
        self._fpeak = None
        self.build_parameter_type(inherits=(E, A, As, B, C))


class BilinearSystem(InputOutputSystem):
    r"""Class for bilinear systems.

    This class describes input-output systems given by

    .. math::
        E x'(t) &= A x(t) + \sum_{i = 1}^m{N_i x(t) u_i(t)} + B u(t) \\
           y(t) &= C x(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + \sum_{i = 1}^m{N_i x(k) u_i(k)} + B u(k) \\
              y(k) &= C x(k)

    if discrete-time, where :math:`E`, :math:`A`, :math:`N_i`, :math:`B`,
    and :math:`C` are linear operators and :math:`m` is the number of
    inputs.

    Parameters
    ----------
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    A
        The |Operator| A.
    N
        The tuple of |Operators| N_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    ss_operators
        A dictonary for state-to-state |Operators| E, A, and N_i.
    is_operators
        A dictonary for input-to-state |Operator| B.
    so_operators
        A dictonary for state-to-output |Operator| C.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.

    Attributes
    ----------
    n
        The order of the system (equal to A.source.dim).
    m
        The number of inputs.
    p
        The number of outputs.
    E
        The |Operator| E. The same as `ss_operators['E']`.
    A
        The |Operator| A. The same as `ss_operators['A']`.
    N
        The tuple of |Operators| N_i. The same as `ss_operators['N']`.
    B
        The |Operator| B. The same as `is_operators['B']`.
    C
        The |Operator| C. The same as `so_operators['C']`.
    """
    linear = False

    def __init__(self, E=None, A=None, N=None, B=None, C=None, ss_operators=None, is_operators=None,
                 so_operators=None, io_operators=None, cont_time=True):
        E = E or ss_operators.get('E')
        A = A or ss_operators['A']
        N = N or ss_operators['N']
        B = B or is_operators['B']
        C = C or so_operators['C']
        assert isinstance(A, OperatorInterface) and A.linear
        assert A.source == A.range
        assert isinstance(B, OperatorInterface) and B.linear
        assert B.range == A.source
        assert isinstance(N, tuple)
        assert len(N) == B.source.dim
        assert all(isinstance(Ni, OperatorInterface) and Ni.linear for Ni in N)
        assert all(Ni.source == Ni.range == A.source for Ni in N)
        assert isinstance(C, OperatorInterface) and C.linear
        assert C.source == A.range
        assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        self.n = A.source.dim
        self.m = B.source.dim
        self.p = C.range.dim
        self.E = E if E is not None else IdentityOperator(A.source)
        self.A = A
        self.N = N
        self.B = B
        self.C = C
        self.ss_operators = FrozenDict({'E': self.E, 'A': A, 'N': N})
        self.is_operators = FrozenDict({'B': B})
        self.so_operators = FrozenDict({'C': C})
        self.io_operators = FrozenDict({})
        self.cont_time = cont_time
        self._poles = None
        self._w = None
        self._tfw = None
        self._gramian = {}
        self._sv = {}
        self._U = {}
        self._V = {}
        self._H2_norm = None
        self._Hinf_norm = None
        self._fpeak = None
        self.build_parameter_type(inherits=(E, A, N, B, C))
