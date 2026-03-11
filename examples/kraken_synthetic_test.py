#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0

if __name__ == '__main__':
    import ctypes, sys
    if sys.platform.startswith('linux'):
        try:
            ctypes.cdll.LoadLibrary('libX11.so').XInitThreads()
        except Exception:
            print("Warning: XInitThreads() failed")

import sys
import signal
import math
import time
import numpy as np
import threading
from PyQt5 import Qt
from gnuradio import gr, analog, blocks, qtgui
from gnuradio.fft import window
import sip

SAMP_RATE = 2_400_000
FFT_SIZE  = 4096

WAVEFORMS = {
    "CW (Sine)":   analog.GR_COS_WAVE,
    "Square":      analog.GR_SQR_WAVE,
    "Sawtooth":    analog.GR_SAW_WAVE,
    "Triangle":    analog.GR_TRI_WAVE,
}

NOISE_TYPES = {
    "Gaussian":  analog.GR_GAUSSIAN,
    "Uniform":   analog.GR_UNIFORM,
    "Laplacian": analog.GR_LAPLACIAN,
    "Impulse":   analog.GR_IMPULSE,
}


# ─────────────────────────────────────────────────────────────
# Custom block: fixed phase rotation
# ─────────────────────────────────────────────────────────────
class PhaseShift(gr.sync_block):
    """Apply a fixed IQ phase rotation."""
    def __init__(self, phase_deg=0.0):
        gr.sync_block.__init__(self, "PhaseShift",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.set_phase(phase_deg)

    def set_phase(self, phase_deg):
        self._phasor = np.exp(1j * math.radians(phase_deg)).astype(np.complex64)

    def work(self, input_items, output_items):
        output_items[0][:] = input_items[0] * self._phasor
        return len(output_items[0])


# ─────────────────────────────────────────────────────────────
# Custom block: IQ phase/power monitor → console + GUI labels
# ─────────────────────────────────────────────────────────────
class ChannelMonitor(gr.sync_block):
    """
    Computes per-update-window:
      - Phase difference  phi = angle(mean(conj(a)*b))
      - Cross-correlation coefficient
      - Per-channel power in dBfs
      - SNR estimate (signal power / noise floor)
    Calls an optional callback(phase, corr, p0_db, p1_db, snr0, snr1).
    """
    UPDATE_N = 65536

    def __init__(self, callback=None):
        gr.sync_block.__init__(self, "ChannelMonitor",
                               in_sig=[np.complex64, np.complex64],
                               out_sig=None)
        self._buf  = [[], []]
        self._lock = threading.Lock()
        self._cb   = callback

    def work(self, input_items, output_items):
        with self._lock:
            self._buf[0].extend(input_items[0].tolist())
            self._buf[1].extend(input_items[1].tolist())

        if len(self._buf[0]) >= self.UPDATE_N:
            with self._lock:
                a = np.array(self._buf[0][:self.UPDATE_N], dtype=np.complex64)
                b = np.array(self._buf[1][:self.UPDATE_N], dtype=np.complex64)
                self._buf[0] = self._buf[0][self.UPDATE_N:]
                self._buf[1] = self._buf[1][self.UPDATE_N:]

            cross = np.mean(np.conj(a) * b)
            phase = float(np.degrees(np.angle(cross)))
            pa    = float(np.mean(np.abs(a) ** 2))
            pb    = float(np.mean(np.abs(b) ** 2))
            corr  = float(np.abs(cross) / (np.sqrt(pa) * np.sqrt(pb) + 1e-12))
            p0_db = float(10 * np.log10(pa + 1e-12))
            p1_db = float(10 * np.log10(pb + 1e-12))

            # Simple SNR estimate via percentile split
            mag_a = np.abs(a)
            noise_floor_a = float(np.percentile(mag_a, 20) ** 2)
            snr0 = float(10 * np.log10((pa - noise_floor_a) / (noise_floor_a + 1e-12) + 1e-12))
            mag_b = np.abs(b)
            noise_floor_b = float(np.percentile(mag_b, 20) ** 2)
            snr1 = float(10 * np.log10((pb - noise_floor_b) / (noise_floor_b + 1e-12) + 1e-12))

            bar_pos = max(0, min(39, int((phase + 180) / 360 * 40)))
            bar = "[" + "·" * bar_pos + "█" + "·" * (39 - bar_pos) + "]"
            print(f"  dPhi={phase:+7.2f}°  {bar}  "
                  f"P0={p0_db:+5.1f} P1={p1_db:+5.1f} dBfs  "
                  f"SNR0={snr0:+5.1f} SNR1={snr1:+5.1f} dB  corr={corr:.3f}")

            if self._cb:
                self._cb(phase, corr, p0_db, p1_db, snr0, snr1)

        return len(input_items[0])


# ─────────────────────────────────────────────────────────────
# Main flow graph
# ─────────────────────────────────────────────────────────────
class KrakenSyntheticTest(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "KrakenSDR Synthetic Test", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("KrakenSDR – Synthetic Signal Test  [no hardware required]")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except Exception:
            pass

        main_layout = Qt.QVBoxLayout()
        self.setLayout(main_layout)
        scroll = Qt.QScrollArea()
        scroll.setFrameStyle(Qt.QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        inner = Qt.QWidget()
        scroll.setWidget(inner)
        self.grid = Qt.QGridLayout(inner)

        self.settings = Qt.QSettings("GNU Radio", "kraken_synthetic_test")
        try:
            self.restoreGeometry(self.settings.value("geometry"))
        except Exception:
            pass

        # ── Signal parameters ────────────────────────────────────
        self.samp_rate    = SAMP_RATE
        self.tone_freq    = 200_000.0
        self.tone_amp     = 0.5
        self.noise_amp    = 0.1
        self.phase_shift  = 45.0
        self.waveform     = analog.GR_COS_WAVE
        self.noise_type   = analog.GR_GAUSSIAN
        self._sweep_active = False
        self._sweep_thread = None

        # ── Row 0: FFT sinks side by side ────────────────────────
        self._sinks = []
        for col, label in enumerate(["CH0  (reference)", "CH1  (phase-shifted)"]):
            sink = qtgui.sink_c(
                FFT_SIZE, window.WIN_BLACKMAN_hARRIS,
                0, self.samp_rate, label,
                True, True, True, False, None
            )
            sink.set_update_time(1.0 / 15)
            sink.enable_rf_freq(False)
            win = sip.wrapinstance(sink.qwidget(), Qt.QWidget)
            self.grid.addWidget(win, 0, col, 1, 1)
            self.grid.setRowStretch(0, 3)
            self.grid.setColumnStretch(col, 1)
            self._sinks.append(sink)

        # ── Row 1: live status bar ───────────────────────────────
        self._status = Qt.QLabel("Waiting for first measurement…")
        self._status.setStyleSheet(
            "background:#0d1b2a; color:#00d4ff; padding:5px; "
            "font-family:monospace; font-size:12px; border-radius:4px;")
        self._status.setAlignment(Qt.Qt.AlignCenter)
        self.grid.addWidget(self._status, 1, 0, 1, 2)
        self.grid.setRowStretch(1, 0)

        # ── Row 2: controls ──────────────────────────────────────
        ctrl = Qt.QGroupBox("Signal Controls")
        ctrl_layout = Qt.QGridLayout(ctrl)

        def make_row(slider, lbl):
            w = Qt.QWidget()
            h = Qt.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(slider)
            lbl.setMinimumWidth(72)
            lbl.setAlignment(Qt.Qt.AlignRight | Qt.Qt.AlignVCenter)
            h.addWidget(lbl)
            return w

        # Tone frequency
        self._tone_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self._tone_slider.setRange(-1_000_000, 1_000_000)
        self._tone_slider.setSingleStep(10_000)
        self._tone_slider.setValue(int(self.tone_freq))
        self._tone_label = Qt.QLabel(f"{self.tone_freq/1e3:+.0f} kHz")
        self._tone_slider.valueChanged.connect(self._on_tone_freq)
        ctrl_layout.addWidget(Qt.QLabel("Tone frequency:"), 0, 0)
        ctrl_layout.addWidget(make_row(self._tone_slider, self._tone_label), 0, 1)

        # Tone amplitude
        self._amp_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self._amp_slider.setRange(0, 100)
        self._amp_slider.setValue(int(self.tone_amp * 100))
        self._amp_label = Qt.QLabel(f"{self.tone_amp:.2f}")
        self._amp_slider.valueChanged.connect(self._on_tone_amp)
        ctrl_layout.addWidget(Qt.QLabel("Tone amplitude (0–1):"), 1, 0)
        ctrl_layout.addWidget(make_row(self._amp_slider, self._amp_label), 1, 1)

        # Noise amplitude
        self._noise_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self._noise_slider.setRange(0, 100)
        self._noise_slider.setValue(int(self.noise_amp * 100))
        self._noise_label = Qt.QLabel(f"{self.noise_amp:.2f}")
        self._noise_slider.valueChanged.connect(self._on_noise_amp)
        ctrl_layout.addWidget(Qt.QLabel("Noise amplitude:"), 2, 0)
        ctrl_layout.addWidget(make_row(self._noise_slider, self._noise_label), 2, 1)

        # Phase shift
        self._phase_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self._phase_slider.setRange(-180, 180)
        self._phase_slider.setValue(int(self.phase_shift))
        self._phase_label = Qt.QLabel(f"{self.phase_shift:+.0f}°")
        self._phase_slider.valueChanged.connect(self._on_phase)
        ctrl_layout.addWidget(Qt.QLabel("CH1 phase shift:"), 3, 0)
        ctrl_layout.addWidget(make_row(self._phase_slider, self._phase_label), 3, 1)

        # Waveform selector
        self._wave_combo = Qt.QComboBox()
        for name in WAVEFORMS:
            self._wave_combo.addItem(name)
        self._wave_combo.currentIndexChanged.connect(self._on_waveform)
        ctrl_layout.addWidget(Qt.QLabel("Waveform:"), 0, 2)
        ctrl_layout.addWidget(self._wave_combo, 0, 3)

        # Noise type selector
        self._noise_combo = Qt.QComboBox()
        for name in NOISE_TYPES:
            self._noise_combo.addItem(name)
        self._noise_combo.currentIndexChanged.connect(self._on_noise_type)
        ctrl_layout.addWidget(Qt.QLabel("Noise type:"), 1, 2)
        ctrl_layout.addWidget(self._noise_combo, 1, 3)

        # Tone ON/OFF toggle
        self._tone_btn = Qt.QPushButton("Tone  ON")
        self._tone_btn.setCheckable(True)
        self._tone_btn.setChecked(True)
        self._tone_btn.setStyleSheet("QPushButton:checked { background: #1a5c1a; color: #7fff7f; }"
                                     "QPushButton { background: #5c1a1a; color: #ff9090; }")
        self._tone_btn.toggled.connect(self._on_tone_toggle)
        ctrl_layout.addWidget(Qt.QLabel("Tone gate:"), 2, 2)
        ctrl_layout.addWidget(self._tone_btn, 2, 3)

        # Frequency sweep button
        self._sweep_btn = Qt.QPushButton("▶  Start Sweep")
        self._sweep_btn.setCheckable(True)
        self._sweep_btn.setChecked(False)
        self._sweep_btn.toggled.connect(self._on_sweep)
        ctrl_layout.addWidget(Qt.QLabel("Freq sweep:"), 3, 2)
        ctrl_layout.addWidget(self._sweep_btn, 3, 3)

        ctrl_layout.setColumnStretch(1, 3)
        ctrl_layout.setColumnStretch(3, 1)
        self.grid.addWidget(ctrl, 2, 0, 1, 2)
        self.grid.setRowStretch(2, 0)

        # ── GNU Radio blocks ─────────────────────────────────────
        self.tone_src  = analog.sig_source_c(
            self.samp_rate, self.waveform, self.tone_freq, self.tone_amp, 0)
        self.noise0    = analog.noise_source_c(self.noise_type, self.noise_amp, 42)
        self.noise1    = analog.noise_source_c(self.noise_type, self.noise_amp, 73)
        self.phase_blk = PhaseShift(self.phase_shift)
        self.add0      = blocks.add_cc()
        self.add1      = blocks.add_cc()
        # Mute valve for the tone (pass_through=True initially)
        self.tone_valve = blocks.copy(gr.sizeof_gr_complex)
        self.tone_valve.set_enabled(True)
        self.null_tone  = blocks.null_source(gr.sizeof_gr_complex)
        self.tone_mux   = blocks.selector(gr.sizeof_gr_complex, 0, 0)  # 2 in, 1 out
        self.chan_mon   = ChannelMonitor(callback=self._on_monitor_update)

        # ── Connections ──────────────────────────────────────────
        #
        #  [tone_src] ──► [phase_blk] ──► [add1 in0]
        #      │                                         [noise1] ──► [add1 in1] ──► [sink1] ──► [mon port1]
        #      └──────────────────────► [add0 in0]
        #                                         [noise0] ──► [add0 in1] ──► [sink0] ──► [mon port0]
        #
        self.connect((self.tone_src,  0), (self.add0,       0))
        self.connect((self.noise0,    0), (self.add0,       1))
        self.connect((self.tone_src,  0), (self.phase_blk,  0))
        self.connect((self.phase_blk, 0), (self.add1,       0))
        self.connect((self.noise1,    0), (self.add1,       1))
        self.connect((self.add0,      0), (self._sinks[0],  0))
        self.connect((self.add1,      0), (self._sinks[1],  0))
        self.connect((self.add0,      0), (self.chan_mon,    0))
        self.connect((self.add1,      0), (self.chan_mon,    1))

    # ── Monitor callback (runs in GR thread, post to Qt via signal) ──
    def _on_monitor_update(self, phase, corr, p0, p1, snr0, snr1):
        bar_pos = max(0, min(39, int((phase + 180) / 360 * 40)))
        bar = "·" * bar_pos + "█" + "·" * (39 - bar_pos)
        txt = (f"  dPhi = {phase:+7.2f}°   [{bar}]   "
               f"P0 = {p0:+5.1f} dBfs   P1 = {p1:+5.1f} dBfs   "
               f"SNR0 = {snr0:+5.1f} dB   SNR1 = {snr1:+5.1f} dB   "
               f"corr = {corr:.3f}")
        # Qt.QMetaObject.invokeMethod to update label from non-Qt thread
        Qt.QMetaObject.invokeMethod(
            self._status, "setText",
            Qt.Qt.QueuedConnection,
            Qt.Q_ARG("QString", txt)
        )

    # ── Slider callbacks ─────────────────────────────────────────
    def _on_tone_freq(self, val):
        self.tone_freq = float(val)
        self._tone_label.setText(f"{val/1e3:+.0f} kHz")
        self.tone_src.set_frequency(self.tone_freq)

    def _on_tone_amp(self, val):
        self.tone_amp = val / 100.0
        self._amp_label.setText(f"{self.tone_amp:.2f}")
        self.tone_src.set_amplitude(self.tone_amp)

    def _on_noise_amp(self, val):
        self.noise_amp = val / 100.0
        self._noise_label.setText(f"{self.noise_amp:.2f}")
        self.noise0.set_amplitude(self.noise_amp)
        self.noise1.set_amplitude(self.noise_amp)

    def _on_phase(self, val):
        self.phase_shift = float(val)
        self._phase_label.setText(f"{val:+.0f}°")
        self.phase_blk.set_phase(self.phase_shift)

    def _on_waveform(self, idx):
        name = list(WAVEFORMS.keys())[idx]
        self.tone_src.set_waveform(WAVEFORMS[name])

    def _on_noise_type(self, idx):
        name = list(NOISE_TYPES.keys())[idx]
        ntype = NOISE_TYPES[name]
        self.noise0.set_noise_type(ntype)
        self.noise1.set_noise_type(ntype)

    def _on_tone_toggle(self, checked):
        self.tone_src.set_amplitude(self.tone_amp if checked else 0.0)
        self._tone_btn.setText("Tone  ON" if checked else "Tone  OFF")

    def _on_sweep(self, checked):
        if checked:
            self._sweep_btn.setText("■  Stop Sweep")
            self._sweep_active = True
            self._sweep_thread = threading.Thread(
                target=self._sweep_worker, daemon=True)
            self._sweep_thread.start()
        else:
            self._sweep_active = False
            self._sweep_btn.setText("▶  Start Sweep")

    def _sweep_worker(self):
        """Sweeps tone frequency from -1 MHz to +1 MHz and back."""
        step  = 20_000
        delay = 0.05
        freq  = -1_000_000
        direction = 1
        while self._sweep_active:
            freq += step * direction
            if freq >= 1_000_000 or freq <= -1_000_000:
                direction *= -1
            self.tone_freq = float(freq)
            self.tone_src.set_frequency(self.tone_freq)
            Qt.QMetaObject.invokeMethod(
                self._tone_slider, "setValue",
                Qt.Qt.QueuedConnection,
                Qt.Q_ARG("int", int(freq))
            )
            time.sleep(delay)

    # ── Window close ─────────────────────────────────────────────
    def closeEvent(self, event):
        self._sweep_active = False
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()


# ─────────────────────────────────────────────────────────────
def main():
    qapp = Qt.QApplication(sys.argv)
    tb = KrakenSyntheticTest()
    tb.start()
    tb.show()

    print("=" * 72)
    print("  KrakenSDR Synthetic Test  |  GNU Radio internal source")
    print("  dPhi should match the 'CH1 phase shift' slider value")
    print("  corr → 1.0 = fully coherent  |  corr → 0.0 = uncorrelated")
    print("=" * 72)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
