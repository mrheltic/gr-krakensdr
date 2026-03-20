#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 KrakenRF Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class krakensdr_correlator(gr.sync_block):
    """
    docstring for block krakensdr_correlator
    """
    def __init__(self, vec_len=2**20, fft_cut=2048, ema_alpha=0.1):
        gr.sync_block.__init__(self,
            name='Correlation Sample and Phase',   # will show up in GRC
            in_sig=[(np.complex64, vec_len), (np.complex64, vec_len)],
            out_sig=[(np.float32, fft_cut), np.float32]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.vec_len = vec_len
        self.fft_cut = fft_cut
        # Complex exponential moving average across CPIs for stable phase readout.
        # Smoothing in the phasor domain avoids ±180° wrap-around artefacts.
        # alpha=0.1 → ~10 CPI time constant; lower values give more smoothing.
        # At short range (< 3 m) multipath is strong: reduce to 0.05 for more
        # stability at the cost of slower tracking.
        self.ema_alpha = ema_alpha
        self._phase_ema = np.complex64(0)
        # Pre-allocate zero-padded input buffers (2×vec_len each) so that work()
        # never calls np.concatenate or np.zeros per CPI.
        # Layout: _x_padd = [ch0 samples | zeros], _y_padd = [zeros | ch1 samples].
        # The zero halves are written once here and never touched again.
        self._x_padd = np.zeros(2 * vec_len, dtype=np.complex64)
        self._y_padd = np.zeros(2 * vec_len, dtype=np.complex64)

    def work(self, input_items, output_items):
        try:
            # Do correlation in the FFT domain and output correlation plot and calculated phase
            # If samples are correlated, the peak will be centered, and is phase is calibrated, it will be near zero.
            N = self.vec_len

            # Fill pre-allocated buffers with slice assignment — no np.empty or
            # np.concatenate, so no per-CPI multi-MB temporary allocations.
            self._x_padd[:N] = input_items[0][0]   # [ch0 | zeros]
            self._y_padd[N:] = input_items[1][0]   # [zeros | ch1]

            x_fft = np.fft.fft(self._x_padd)
            y_fft = np.fft.fft(self._y_padd)

            x_corr = np.fft.ifft(x_fft.conj() * y_fft)
            x_corr_plot = 10*np.log10(np.abs(x_corr))
            x_corr_plot -= np.max(x_corr_plot)

            M = self.fft_cut // 2
            # .copy() releases the full 2N-element backing array (≈2 MB) immediately
            # rather than keeping it alive via a view until the next CPI.
            x_corr_plot = x_corr_plot[N-M:N+M].copy()

            # Smooth x_corr[N] (zero-lag phasor) across CPIs, then extract angle.
            # This is equivalent to coherently averaging multiple CPIs before computing
            # the phase, which greatly reduces noise without introducing wrap-around bias.
            self._phase_ema = ((1.0 - self.ema_alpha) * self._phase_ema
                               + self.ema_alpha * x_corr[N])
            phase = np.rad2deg(np.angle(self._phase_ema))

            if not np.isnan(x_corr_plot).any():
                output_items[0][:] = x_corr_plot
                output_items[1][:] = phase
        except:
            pass

        return len(output_items[0])
