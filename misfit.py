import numpy as np
import pylops
import core


class MisfitFunction:
    r"""General misfit function

        Maps model parameters m to misfit values:
        R^{N} -> R
        N - total number of model parameters

        Parameters
        ----------
        dataMisfit : :class:
            DataMisfit class
        regMisfit : :class:
            RegMisfit class
    """

    def __init__(self, dataMisfit, regMisfit):
        self.data_misfit = dataMisfit
        self.reg_misfit = regMisfit

    def evaluate(self, m):
        """Evaluate the misfit function

        Data misfit + Regularization misfit

        Parameters
        ----------
        m : :obj:`numpy.ndarray`
            Vector of model parameters (slowness-squared model)

        Returns
        -------
        f : :obj:`int`
            regularized misfit value at m
        g : :obj:`numpy.ndarray`
            Vector of regularized misfit gradient
        H : :obj:`pylops.LinearOperator`
            Regularized misfit Hessian operator
        """

        fd, gd, Hd = self.data_misfit.evaluate(m)
        fr, gr, Hr = self.reg_misfit.evaluate(m)

        f = fd + fr
        g = gd + gr
        H = Hd + Hr
        return f, g, H


class DataMisfit:
    r"""Least-squares data misfit function

        Maps model parameters m to misfit values:
        R^{N} -> R
        N - total number of model parameters

        Parameters
        ----------
        Dobs : :obj:`numpy.ndarray`
            1-D flattened array of observed data (Nr x Ns x Nf)
            Nr - total number of receivers
            Ns - total number of sources
            Nf - total number of frequencies
        model : :obj:`dict`
            Dictionary of parameters for modelling
    """

    def __init__(self, Dobs, model):
        self.Fm = core.ForwardSolver(model)
        self.Dobs = Dobs

    def evaluate(self, m):
        """Evaluate the misfit function

        0.5||P.T*A^{-1}(m)Q - Dobs||_{F}^2

        where P and Q encode the receiver and source locations

        Parameters
        ----------
        m : :obj:`numpy.ndarray`
            Vector of model parameters (slowness-squared model)

        Returns
        -------
        f : :obj:`int`
            misfit value at m
        g : :obj:`numpy.ndarray`
            Vector of data misfit gradient
        H : :obj:`pylops.LinearOperator`
            Data misfit Hessian operator
        """
        Dk, Jk = self.Fm.solve(m)

        f = 0.5 * np.linalg.norm(Dk - self.Dobs, 'fro') ** 2
        g = Jk.H * (Dk - self.Dobs)
        H = Jk.H * Jk

        return f, g.real.reshape(-1, 1), H


class RegMisfit:
    r"""Regularization misfit function

        Maps model parameters m to misfit values:
        R^{N} -> R
        N - total number of model parameters

        Parameters
        ----------
        n : :obj:`numpy.ndarray`
            Dimensions of m (nz,nx)
        alpha : :obj:`int`
            Regularization weights
        m0 : :obj:`numpy.ndarray`
            Vector of model parameters prior (size of m)
    """

    def __init__(self, n, alpha=0.5, m0=0.0):
        self.m0 = m0
        self.alpha = alpha
        self.L = pylops.FirstDerivative(np.prod(n), edge=True)

    def evaluate(self, m):
        """Evaluate the misfit function

        \alpha||L*(m - m0)||_{F}^2

        where L is the 1st derivative operators

        Parameters
        ----------
        m : :obj:`numpy.ndarray`
            Vector of model parameters (slowness-squared model)

        Returns
        -------
        f : :obj:`int`
            misfit value at m
        g : :obj:`numpy.ndarray`
            Vector of data misfit gradient
        H : :obj:`pylops.LinearOperator`
            Data misfit Hessian operator
        """

        LL = self.L.T*self.L

        f = self.alpha*np.linalg.norm(self.L*(m - self.m0), 'fro')**2
        g = self.alpha * LL * (m - self.m0)
        H = self.alpha * LL

        return f, g.real.reshape(-1, 1), H
