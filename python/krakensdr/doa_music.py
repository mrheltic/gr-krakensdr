#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 KrakenRF Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
import numpy.linalg as lin
#from scipy import signal

from gnuradio import gr

class doa_music(gr.sync_block):
    """
    docstring for block doa_music
    """
    def __init__(self, vec_len=1048576, freq=433.0, array_dist=0.33, num_elements=5, array_type='UCA', signal_dimension=1):
        gr.sync_block.__init__(self,
            name="DOA MUSIC",
            in_sig=[(np.complex64, vec_len)] * num_elements,
            out_sig=[(np.float32, 360)])
            
        self.cpi_size = vec_len
        self.freq = freq
        self.array_dist = array_dist
        self.num_elements = num_elements
        self.array_type = array_type
        self.signal_dimension = signal_dimension

        wavelength = 300 / freq
        if array_type == 'UCA':
            inter_elem_spacing = (np.sqrt(2) * array_dist * np.sqrt(1 - np.cos(np.deg2rad(360 / num_elements))))
            wavelength_mult = inter_elem_spacing / wavelength
        else:
            wavelength_mult = array_dist / wavelength
        
        self.scanning_vectors = self.gen_scanning_vectors(self.num_elements, wavelength_mult, self.array_type, 0)

        print("wavelength mult: " + str(wavelength_mult))

        # Pre-allocate the signal matrix reused every CPI to avoid per-call allocation.
        self._processed_signal = np.empty((num_elements, vec_len), dtype=np.complex64)

    # ── Private helper ──────────────────────────────────────────────────────
    def _calc_wavelength_mult(self):
        """Return the normalised inter-element spacing d/λ."""
        wavelength = 300.0 / self.freq  # freq in MHz → λ in meters
        if self.array_type == 'UCA':
            inter_elem_spacing = (np.sqrt(2) * self.array_dist
                                  * np.sqrt(1 - np.cos(np.deg2rad(360.0 / self.num_elements))))
            return inter_elem_spacing / wavelength
        else:  # ULA
            return self.array_dist / wavelength

    def _rebuild_scanning_vectors(self):
        wavelength_mult = self._calc_wavelength_mult()
        self.scanning_vectors = self.gen_scanning_vectors(
            self.num_elements, wavelength_mult, self.array_type, 0)

    # ── Runtime setters (called by GNU Radio flowgraph variable callbacks) ──
    def set_freq(self, freq):
        """Update the centre frequency [MHz] and recompute scanning vectors."""
        self.freq = freq
        self._rebuild_scanning_vectors()

    def set_array_dist(self, array_dist):
        """Update the antenna spacing [m] and recompute scanning vectors."""
        self.array_dist = array_dist
        self._rebuild_scanning_vectors()

    def set_signal_dimension(self, signal_dimension):
        """Update the number of expected coherent sources (1 = single dominant source)."""
        self.signal_dimension = max(1, min(int(signal_dimension), self.num_elements - 1))
        
    def work(self, input_items, output_items):

        #print("input items size: " + str(np.shape(input_items)))
        # Reuse pre-allocated matrix; avoids a fresh np.empty each CPI.
        for i in range(self.num_elements):
            self._processed_signal[i, :] = input_items[i][0]

        #decimated_processed_signal = signal.decimate(processed_signal, 100, n=100 * 2, ftype='fir')
        # Doing decimation in GNU Radio blocks, or uncomment to do decimation in scipy
        decimated_processed_signal = self._processed_signal
       
        R = self.corr_matrix(decimated_processed_signal)
        DOA_MUSIC_res = self.DOA_MUSIC(R, self.scanning_vectors, signal_dimension=self.signal_dimension)
        doa_plot = self.DOA_plot_util(DOA_MUSIC_res)
        output_items[0][0][:] = doa_plot
                       
        return len(output_items[0])


    def corr_matrix(self, X):
        N = X[0, :].size
        R = np.dot(X, X.conj().T)
        R = np.divide(R, N)
        # Forward-backward spatial averaging: decorrelates coherent multipath.
        # For centrosymmetric arrays (UCA, ULA) the exchange matrix J reverses
        # element order; R_fb has the same noise subspace but reduced coherence
        # between direct path and reflections.  Cost: one matrix multiply + conj.
        M = R.shape[0]
        J = np.eye(M, dtype=R.dtype)[::-1]   # anti-identity (exchange matrix)
        R = 0.5 * (R + J @ R.conj() @ J)
        return R


    def gen_scanning_vectors(self, M, DOA_inter_elem_space, type, offset):
        thetas = np.linspace(0, 359, 360)  # Remember to change self.DOA_thetas too, we didn't include that in this function due to memoization cannot work with arrays
        if type == "UCA":
            x = DOA_inter_elem_space * np.cos(2 * np.pi / M * np.arange(M))
            y = -DOA_inter_elem_space * np.sin(2 * np.pi / M * np.arange(M))
        elif "ULA":
            x = np.zeros(M)
            y = -np.arange(M) * DOA_inter_elem_space

        scanning_vectors = np.zeros((M, thetas.size), dtype=np.complex64)
        for i in range(thetas.size):
            scanning_vectors[:, i] = np.exp(
                1j * 2 * np.pi * (x * np.cos(np.deg2rad(thetas[i] + offset)) + y * np.sin(np.deg2rad(thetas[i] + offset))))

        return np.ascontiguousarray(scanning_vectors)
        
        
    def DOA_MUSIC(self, R, scanning_vectors, signal_dimension, angle_resolution=1):
        # --> Input check
        if R[:, 0].size != R[0, :].size:
            print("ERROR: Correlation matrix is not quadratic")
            return np.ones(1, dtype=np.complex64) * -1  # [(-1, -1j)]

        if R[:, 0].size != scanning_vectors[:, 0].size:
            print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
            return np.ones(1, dtype=np.complex64) * -2

        ADORT = np.zeros(scanning_vectors[0, :].size, dtype=np.complex64)
        M = R[:, 0].size  # np.size(R, 0)

        # --- Calculation ---
        # Determine eigenvectors and eigenvalues
        sigmai, vi = lin.eig(R)
        sigmai = np.abs(sigmai)

        idx = sigmai.argsort()[::1]  # Sort eigenvectors by eigenvalues, smallest to largest
        vi = vi[:, idx]

        # Generate noise subspace matrix
        noise_dimension = M - signal_dimension

        E = np.zeros((M, noise_dimension), dtype=np.complex64)
        for i in range(noise_dimension):
            E[:, i] = vi[:, i]

        E_ct = E @ E.conj().T
        theta_index = 0
        for i in range(scanning_vectors[0, :].size):
            S_theta_ = scanning_vectors[:, i]
            S_theta_ = np.ascontiguousarray(S_theta_.T)
            ADORT[theta_index] = 1 / np.abs(S_theta_.conj().T @ E_ct @ S_theta_)
            theta_index += 1

        return ADORT
        
    def DOA_plot_util(self, DOA_data, log_scale_min=-100):
        """
            This function prepares the calulcated DoA estimation results for plotting.
            - Noramlize DoA estimation results
            - Changes to log scale
        """

        DOA_data = np.divide(np.abs(DOA_data), np.max(np.abs(DOA_data)))  # Normalization
        DOA_data = 10 * np.log10(DOA_data)  # Change to logscale

        for i in range(len(DOA_data)):  # Remove extremely low values
            if DOA_data[i] < log_scale_min:
                DOA_data[i] = log_scale_min

        return DOA_data


# =============================================================================
#  music_doa_block
#  A self-contained MUSIC DoA estimator for Uniform Circular Arrays (UCA).
#
#  Input:  M streams of np.complex64 IQ samples (one per antenna).
#  Output: 1 stream of np.float32, one value per CPI = the peak bearing [deg].
#
#  The block is a decimator: it consumes `num_snapshots` input samples per
#  output sample, so the output rate = input_rate / num_snapshots.
#
#  Steering vector formula (UCA):
#    a(θ) = exp( j·2π·r·cos(θ − φ_m) )   m = 0..M-1
#  where r is the array radius in units of wavelength, and
#        φ_m = 2π·m/M is the angular position of the m-th element.
# =============================================================================
class music_doa_block(gr.decim_block):
    """
    MUSIC DoA estimator for a Uniform Circular Array (UCA).
    Accepts M complex IQ input streams and outputs one bearing angle per CPI.
    """

    def __init__(self, num_antennas=3, radius_lambda=0.5,
                 num_snapshots=1024, num_signals=1):
        gr.decim_block.__init__(
            self,
            name='MUSIC DoA Estimator',
            in_sig=[np.complex64] * num_antennas,
            out_sig=[np.float32],
            decimation=num_snapshots,
        )

        self.M = num_antennas
        self.r = radius_lambda
        self.D = max(1, min(int(num_signals), num_antennas - 1))

        # Angular search grid: 0..359 degrees, 1-degree resolution
        self.theta_range = np.arange(0, 360, 1)
        theta_rad = np.radians(self.theta_range)

        # Angular positions of the M antenna elements on the circle
        phi_m = np.arange(self.M) * 2.0 * np.pi / self.M   # shape (M,)

        # Pre-compute the steering matrix A_search: shape (M, 360)
        # a[:,k] = steering vector for look-angle theta_range[k]
        phase = 2.0 * np.pi * self.r * np.cos(
            theta_rad[np.newaxis, :] - phi_m[:, np.newaxis])  # (M, 360)
        self.A_search = np.exp(1j * phase).astype(np.complex64)

    # ── runtime setters ───────────────────────────────────────────────────────
    def set_radius_lambda(self, r):
        """Update array radius [wavelengths] and rebuild steering matrix."""
        self.r = r
        phi_m = np.arange(self.M) * 2.0 * np.pi / self.M
        theta_rad = np.radians(self.theta_range)
        phase = 2.0 * np.pi * self.r * np.cos(
            theta_rad[np.newaxis, :] - phi_m[:, np.newaxis])
        self.A_search = np.exp(1j * phase).astype(np.complex64)

    def set_num_signals(self, d):
        """Update the number of expected coherent sources."""
        self.D = max(1, min(int(d), self.M - 1))

    # ── GNU Radio work() ─────────────────────────────────────────────────────
    def work(self, input_items, output_items):
        n_out = len(output_items[0])
        N = self.decimation()   # snapshots per CPI

        for i in range(n_out):
            # Build snapshot matrix X: shape (M, N)
            X = np.empty((self.M, N), dtype=np.complex64)
            for m in range(self.M):
                X[m, :] = input_items[m][i * N: (i + 1) * N]

            # 1. Sample covariance matrix R = X·X^H / N
            R = (X @ X.conj().T) / N

            # 2. Eigendecomposition (eigh: guaranteed real eigenvalues for
            #    Hermitian matrices, returns eigenvectors sorted ascending)
            _, eigvecs = np.linalg.eigh(R)

            # 3. Noise subspace: first (M - D) eigenvectors (smallest eigenvalues)
            Un = eigvecs[:, : self.M - self.D]          # (M, M-D)
            Un_UnH = Un @ Un.conj().T                   # (M, M)

            # 4. MUSIC pseudo-spectrum (vectorised over all angles at once)
            #    P[k] = 1 / |a_k^H · Un_UnH · a_k|
            #    a_k^H · Un_UnH · a_k  = sum_j |( Un^H · a_k )[j]|^2
            proj = Un.conj().T @ self.A_search          # (M-D, 360)
            den = np.sum(np.abs(proj) ** 2, axis=0)     # (360,)
            P_music = 1.0 / (den + 1e-12)

            # 5. Peak bearing
            output_items[0][i] = float(self.theta_range[np.argmax(P_music)])

        return n_out
