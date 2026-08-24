"""
Utilities for computing analytical Poisson posteriors on a parameter grid.

This module defines `PoissonPosteriorGrid`, which builds or loads grids of
expected event rates (lambda) across log10(m_chi) and log10(c_p), then evaluates
analytical Poisson posteriors for total event counts or binned data. It also
includes convenience plotting helpers for visualizing the posterior surfaces.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from configs.config import PARAM_RANGES

class PoissonPosteriorGrid:
    """
    Class for computing and storing grids of expected event rates (λ) and
    analytical Poisson posteriors for WIMP direct detection.

    Parameters
    ----------
    datatag : str
        Qualitative region identifier ("low","mid","high").
    gridbins : int
        Number of grid points for log10(mχ) and log10(c_p).
    mc_generator : callable
        Function (mchi, cp, poisson=False) → λ per recoil bin.
    """

    def __init__(self, datatag, gridbins, mc_generator):
        if datatag not in PARAM_RANGES:
            raise ValueError("level must be one of ['low','mid','high']")

        self.datatag = datatag
        self.gridbins = gridbins
        self.mc_generator = mc_generator

        self.logm_range = PARAM_RANGES[datatag]["logm_range"]
        self.logcp_range = PARAM_RANGES[datatag]["logcp_range"]

        self.lambda_grid = None
        self.logm_vals = None
        self.logcp_vals = None

    def _filename(self):
        return f"lambda_{self.datatag}_bins{self.gridbins}.npz"

    def _lambda_path(self):
        return os.path.join("analytical", "lambda", self._filename())

    def _ensure_grid_loaded(self):
        if self.lambda_grid is None or self.logm_vals is None or self.logcp_vals is None:
            raise RuntimeError("lambda grid not initialized; call load_or_compute() first")

    def _grid_spacing(self):
        dx = self.logm_vals[1] - self.logm_vals[0]
        dy = self.logcp_vals[1] - self.logcp_vals[0]
        return dx, dy

    def _log_grids(self):
        return np.meshgrid(self.logcp_vals, self.logm_vals, indexing="ij")

    def compute_lambda_grid(self):
        self.logm_vals = np.linspace(*self.logm_range, self.gridbins)
        self.logcp_vals = np.linspace(*self.logcp_range, self.gridbins)

        logcp_grid, logm_grid = self._log_grids()
        mchi_grid, cp_grid = 10**logm_grid, 10**logcp_grid

        lambda_grid = np.empty(mchi_grid.shape, dtype=object)

        for i in range(mchi_grid.shape[0]):
            for j in range(mchi_grid.shape[1]):
                _, expected_events, _, _ = self.mc_generator(mchi_grid[i, j], cp_grid[i, j], poisson=False)
                lambda_grid[i, j] = expected_events

        self.lambda_grid = lambda_grid

    def save_lambda(self):
        np.savez(
            self._lambda_path(),
            lambda_grid=self.lambda_grid,
            logm_vals=self.logm_vals,
            logcp_vals=self.logcp_vals,
        )

    @classmethod
    def load_or_compute(cls, datatag, gridbins, mc_generator):
        obj = cls(datatag, gridbins, mc_generator)
        lambda_path = obj._lambda_path()

        if os.path.exists(lambda_path):
            data = np.load(lambda_path, allow_pickle=True)
            obj.lambda_grid = data["lambda_grid"]
            obj.logm_vals  = data["logm_vals"]
            obj.logcp_vals = data["logcp_vals"]
            #print(f"Loaded: analytical/lambda/{filename}")
        else:
            print(f"Generating and saving grid -> {lambda_path}")
            obj.compute_lambda_grid()
            obj.save_lambda()

        return obj

    def posterior_nevents(self, n_observed):
        self._ensure_grid_loaded()
        lambda_tot = np.array([[lam.sum() for lam in row] for row in self.lambda_grid])

        with np.errstate(divide='ignore', invalid='ignore'):
            logL = n_observed * np.log(lambda_tot) - lambda_tot

        logL = np.where((lambda_tot == 0) & (n_observed > 0), -np.inf, logL)
        logL = np.where((lambda_tot == 0) & (n_observed == 0), 0.0, logL)

        logL -= np.max(logL)
        P = np.exp(logL)

        dx, dy = self._grid_spacing()
        return P / (P.sum() * dx * dy)

    def posterior_binned(self, n_obs_bin):
        self._ensure_grid_loaded()
        P = np.zeros(self.lambda_grid.shape)

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                lam = self.lambda_grid[i,j]
                if np.any((lam == 0) & (n_obs_bin > 0)):
                    logL = -np.inf
                else:
                    lam_safe = np.where(lam==0,1.0,lam)
                    logL = np.sum(n_obs_bin * np.log(lam_safe) - lam)

                P[i,j] = logL

        P -= np.max(P)
        P = np.exp(P)

        dx, dy = self._grid_spacing()
        return P / (P.sum() * dx * dy)

    def plot_posterior_nevents(self, n_observed, levels=25, cmap="viridis"):

        posterior = self.posterior_nevents(n_observed)
        logcp_grid, logm_grid = self._log_grids()

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(10**logm_grid, 10**logcp_grid, posterior, levels=levels, cmap=cmap)
        plt.colorbar(contour, label="Posterior")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$m_\chi\ [\mathrm{GeV}]$")
        plt.ylabel(r"$c_p\ [\mathrm{GeV}^{-2}]$")
        plt.title(rf"Analytical Posterior for N = {n_observed}")
        plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_posterior_binned(self, n_observed_per_bin, mchi_true, cp_true, levels=25, cmap="viridis"):

        posterior = self.posterior_binned(n_observed_per_bin)
        logcp_grid, logm_grid = self._log_grids()

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(10**logm_grid, 10**logcp_grid, posterior, levels=levels, cmap=cmap)
        plt.colorbar(contour, label="Posterior")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$m_\chi\ [\mathrm{GeV}]$")
        plt.ylabel(r"$c_p\ [\mathrm{GeV}^{-2}]$")
        plt.title(rf"Analytical Posterior for binned data")
        plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)
        plt.scatter(mchi_true, cp_true, color='red', s=40, label="True parameters")
        plt.tight_layout()
        plt.show()

