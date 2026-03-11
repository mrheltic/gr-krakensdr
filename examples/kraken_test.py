#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KrakenSDR – Test rapido con antenne corte (10-20 cm)
=====================================================

Con antenne da 10-20 cm sei risonante in λ/4 circa tra 375 e 750 MHz.
Anche fuori risonanza l'RTL-SDR riceve comunque, con sensibilità minore.

Frequenze consigliate e cosa aspettarsi:

  [433.9 MHz]  ISM 433 – OTTIMO con antenne corte (~17 cm = λ/4)
               Vedrai burst brevi e intensi ogni volta che qualcuno
               preme un telecomando, un sensore meteo trasmette, un
               allarme auto invia un segnale, ecc.

  [868.0 MHz]  ISM 868 (EU) – OTTIMO con 8-9 cm
               LoRa, contatori smart, sensori industriali, Zigbee.
               Segnali più rari ma ben visibili nel waterfall.

  [446.0 MHz]  PMR446 – walkie-talkie civili senza licenza
               Se qualcuno parla nelle vicinanze lo vedi subito come
               un segnale FM largo ~12.5 kHz.

  [1090.0 MHz] ADS-B – transponder aerei
               Impulsi brevi e potenti, visibili anche con antenne
               non ottimizzate. Se sei in zona con traffico aereo
               vedrai burst continui nel waterfall.

  [100.0 MHz]  Radio FM – segnale fortissimo ma antenne troppo corte
               Lo vedi comunque grazie alla potenza, ma distorto.

  [462.5 MHz]  Bande varie (dipende dalla zona)

Uso: python3 kraken_test.py
"""

if __name__ == '__main__':
    import ctypes, ctypes.util, sys
    if sys.platform.startswith('linux'):
        try:
            # Must use find_library: Ubuntu 22.04 ships 'libX11.so.6' with no
            # bare 'libX11.so' symlink unless libx11-dev is installed.
            _libX11 = ctypes.util.find_library('X11')
            if _libX11:
                ctypes.cdll.LoadLibrary(_libX11).XInitThreads()
            else:
                print("Warning: libX11 not found — XInitThreads() skipped")
        except Exception as _e:
            print(f"Warning: XInitThreads() failed: {_e}")

import sys, signal
from PyQt5 import Qt
from gnuradio import gr, eng_notation, qtgui
from gnuradio.fft import window
import sip
from gnuradio import krakensdr

# ─────────────────────────────────────────────────────────────
# Preset: (etichetta, freq_MHz, nota)
# ─────────────────────────────────────────────────────────────
PRESETS = [
    ("433.9 MHz – ISM 433 (remotes/sensors)",         433.9,  "Short bursts: remotes, alarms, weather sensors"),
    ("868.0 MHz – ISM 868 / LoRa (EU)",               868.0,  "LoRa, smart meters, industrial sensors"),
    ("446.0 MHz – PMR446 (walkie-talkie)",             446.0,  "FM signal if someone is talking nearby"),
    ("1090.0 MHz – ADS-B (aircraft transponders)",   1090.0,  "Continuous bursts if there is air traffic"),
    ("462.5 MHz – FRS/misc",                           462.5,  "Mixed band, depends on local usage"),
    ("100.0 MHz – FM Radio (low sensitivity)",         100.0,  "Very strong signal, short antennas but it works"),
    ("144.8 MHz – APRS (amateur radio)",               144.8,  "GPS packets from amateur radio operators and vehicles"),
    ("406.0 MHz – EPIRB/ELT (emergency beacons)",     406.0,  "Rare, visible during tests or false alarms"),
]

SAMP_RATE   = 2_400_000   # 2.4 MHz di larghezza di banda
NUM_CHANNELS = 2
FFT_SIZE    = 4096


class KrakenTest(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "KrakenSDR Test", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("KrakenSDR – Short Antenna Test")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except Exception:
            pass

        # ── Layout ──────────────────────────────────────────────
        main_layout = Qt.QVBoxLayout()
        self.setLayout(main_layout)

        scroll = Qt.QScrollArea()
        scroll.setFrameStyle(Qt.QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        inner = Qt.QWidget()
        scroll.setWidget(inner)
        self.grid = Qt.QGridLayout(inner)

        self.settings = Qt.QSettings("GNU Radio", "kraken_test")
        try:
            self.restoreGeometry(self.settings.value("geometry"))
        except Exception:
            pass

        # ── Parametri ───────────────────────────────────────────
        self.samp_rate = SAMP_RATE
        self.gain      = 40.2
        self.freq      = PRESETS[0][1]

        # ── Riga 0: barra strumenti ─────────────────────────────
        tb = Qt.QToolBar(self)
        tb.setMovable(False)

        tb.addWidget(Qt.QLabel("  Preset:  "))
        self._combo = Qt.QComboBox()
        self._combo.setMinimumWidth(380)
        for label, _, _ in PRESETS:
            self._combo.addItem(label)
        self._combo.currentIndexChanged.connect(self._on_preset)
        tb.addWidget(self._combo)

        tb.addWidget(Qt.QLabel("   Freq [MHz]: "))
        self._freq_edit = Qt.QLineEdit(str(self.freq))
        self._freq_edit.setMaximumWidth(80)
        self._freq_edit.editingFinished.connect(
            lambda: self.set_freq(float(self._freq_edit.text())))
        tb.addWidget(self._freq_edit)

        tb.addWidget(Qt.QLabel("   Gain [dB]: "))
        self._gain_edit = Qt.QLineEdit(str(self.gain))
        self._gain_edit.setMaximumWidth(55)
        self._gain_edit.editingFinished.connect(
            lambda: self.set_gain(float(self._gain_edit.text())))
        tb.addWidget(self._gain_edit)

        self.grid.addWidget(tb, 0, 0, 1, 2)

        # ── Riga 1: etichetta suggerimento ─────────────────────
        self._hint_label = Qt.QLabel()
        self._hint_label.setStyleSheet(
            "background:#1e3a5f; color:#7ecfff; padding:6px; font-size:12px;")
        self._hint_label.setWordWrap(True)
        self._hint_label.setText(f"ℹ️  {PRESETS[0][2]}")
        self.grid.addWidget(self._hint_label, 1, 0, 1, 2)

        # ── Riga 2-3: waterfall ch0 e ch1 ──────────────────────
        self._sinks = []
        for ch in range(NUM_CHANNELS):
            sink = qtgui.sink_c(
                FFT_SIZE,
                window.WIN_BLACKMAN_hARRIS,
                self.freq,
                self.samp_rate,
                f"Channel {ch}",
                True,   # FFT
                True,   # Waterfall
                False,  # Tempo
                False,  # Costellazione
                None
            )
            sink.set_update_time(1.0 / 15)
            sink.enable_rf_freq(True)
            win = sip.wrapinstance(sink.qwidget(), Qt.QWidget)
            self.grid.addWidget(win, 2 + ch, 0, 1, 2)
            self.grid.setRowStretch(2 + ch, 1)
            self._sinks.append(sink)

        # ── Sorgente KrakenSDR ──────────────────────────────────
        self.src = krakensdr.krakensdr_source(
            '127.0.0.1', 5000, 5001,
            NUM_CHANNELS,
            self.freq,
            [self.gain] * NUM_CHANNELS,
            False
        )

        # ── Connessioni ─────────────────────────────────────────
        for ch in range(NUM_CHANNELS):
            self.connect((self.src, ch), (self._sinks[ch], 0))

    # ── Callback combo preset ───────────────────────────────────
    def _on_preset(self, idx):
        _, freq, hint = PRESETS[idx]
        self._hint_label.setText(f"ℹ️  {hint}")
        self.set_freq(freq)

    # ── Chiusura ────────────────────────────────────────────────
    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    # ── Setter freq ─────────────────────────────────────────────
    def set_freq(self, freq):
        self.freq = freq
        self._freq_edit.setText(str(freq))
        self.src.set_freq(self.freq)
        for sink in self._sinks:
            sink.set_frequency_range(self.freq, self.samp_rate)
        print(f"[→] Frequency: {freq} MHz  |  BW: {self.samp_rate/1e6:.1f} MHz")

    # ── Setter gain ─────────────────────────────────────────────
    def set_gain(self, gain):
        self.gain = gain
        self._gain_edit.setText(str(gain))
        self.src.set_gain([gain] * NUM_CHANNELS)
        print(f"[→] Gain: {gain} dB")


# ─────────────────────────────────────────────────────────────
def main():
    qapp = Qt.QApplication(sys.argv)
    tb = KrakenTest()
    tb.start()
    tb.show()

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
