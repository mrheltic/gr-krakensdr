#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# KrakenSDR FFT display - 2 canali
# Adatto quando si hanno solo 2 antenne (ch0 e ch1).
#
# NOTA: RTL-SDR copre ~24 MHz – 1766 MHz
# Frequenze consigliate per sperimentare:
#   100.0   -> Radio FM
#   137.5   -> Satelliti NOAA meteo
#   433.9   -> IoT / telecomandi
#   1090.0  -> ADS-B aerei  (ottimo!)
#   1575.42 -> GPS L1
#
# SPDX-License-Identifier: GPL-3.0

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import eng_notation
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import krakensdr

from gnuradio import qtgui

class kraken_fft_2ch(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "KrakenSDR FFT - 2 Canali", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("KrakenSDR FFT - 2 Canali")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "kraken_fft_2ch")

        try:
            self.restoreGeometry(self.settings.value("geometry"))
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variabili
        ##################################################
        self.samp_rate = samp_rate = 2400000   # Larghezza di banda: 2.4 MHz
        self.gain      = gain      = 40.2      # Guadagno IF [dB] (0 – 49.6)
        self.freq      = freq      = 433.9     # Frequenza centrale [MHz] — modifica qui!
        self.num_ch    = num_ch    = 2         # Numero di canali attivi

        ##################################################
        # Barra strumenti: Frequenza
        ##################################################
        self._freq_tool_bar = Qt.QToolBar(self)
        self._freq_tool_bar.addWidget(Qt.QLabel("Frequenza centrale [MHz]: "))
        self._freq_line_edit = Qt.QLineEdit(str(self.freq))
        self._freq_tool_bar.addWidget(self._freq_line_edit)
        self._freq_line_edit.editingFinished.connect(
            lambda: self.set_freq(eng_notation.str_to_num(str(self._freq_line_edit.text()))))
        self.top_grid_layout.addWidget(self._freq_tool_bar, 0, 0, 1, 1)

        ##################################################
        # Barra strumenti: Guadagno
        ##################################################
        self._gain_tool_bar = Qt.QToolBar(self)
        self._gain_tool_bar.addWidget(Qt.QLabel("Guadagno [0 – 49.6 dB]: "))
        self._gain_line_edit = Qt.QLineEdit(str(self.gain))
        self._gain_tool_bar.addWidget(self._gain_line_edit)
        self._gain_line_edit.editingFinished.connect(
            lambda: self.set_gain(eng_notation.str_to_num(str(self._gain_line_edit.text()))))
        self.top_grid_layout.addWidget(self._gain_tool_bar, 0, 1, 1, 1)

        ##################################################
        # Sink FFT — Canale 0
        ##################################################
        self.qtgui_sink_ch0 = qtgui.sink_c(
            16384,                        # FFT size
            window.WIN_BLACKMAN_hARRIS,   # Finestra
            freq,                         # Freq. centrale [Hz] (viene aggiornata)
            samp_rate,                    # Larghezza di banda
            'Canale 0',                   # Nome
            True,   # FFT (frequenza)
            True,   # Waterfall
            True,   # Dominio del tempo
            False,  # Costellazione
            None
        )
        self.qtgui_sink_ch0.set_update_time(1.0 / 10)
        self.qtgui_sink_ch0.enable_rf_freq(True)
        self._qtgui_sink_ch0_win = sip.wrapinstance(self.qtgui_sink_ch0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_sink_ch0_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)

        ##################################################
        # Sink FFT — Canale 1
        ##################################################
        self.qtgui_sink_ch1 = qtgui.sink_c(
            16384,
            window.WIN_BLACKMAN_hARRIS,
            freq,
            samp_rate,
            'Canale 1',
            True,   # FFT
            True,   # Waterfall
            True,   # Tempo
            False,  # Costellazione
            None
        )
        self.qtgui_sink_ch1.set_update_time(1.0 / 10)
        self.qtgui_sink_ch1.enable_rf_freq(True)
        self._qtgui_sink_ch1_win = sip.wrapinstance(self.qtgui_sink_ch1.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_sink_ch1_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)

        ##################################################
        # Sorgente KrakenSDR — solo 2 canali
        ##################################################
        self.krakensdr_source = krakensdr.krakensdr_source(
            '127.0.0.1',        # IP Heimdall
            5000,               # Porta dati
            5001,               # Porta controllo
            self.num_ch,        # Numero canali (2)
            freq,               # Frequenza [MHz]
            [gain, gain],       # Guadagno per canale
            False               # Debug
        )

        ##################################################
        # Connessioni
        ##################################################
        self.connect((self.krakensdr_source, 0), (self.qtgui_sink_ch0, 0))
        self.connect((self.krakensdr_source, 1), (self.qtgui_sink_ch1, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "kraken_fft_2ch")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    # --- Getter / Setter ---

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_sink_ch0.set_frequency_range(self.freq, self.samp_rate)
        self.qtgui_sink_ch1.set_frequency_range(self.freq, self.samp_rate)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        Qt.QMetaObject.invokeMethod(self._gain_line_edit, "setText",
            Qt.Q_ARG("QString", eng_notation.num_to_str(self.gain)))
        self.krakensdr_source.set_gain([self.gain, self.gain])

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        Qt.QMetaObject.invokeMethod(self._freq_line_edit, "setText",
            Qt.Q_ARG("QString", eng_notation.num_to_str(self.freq)))
        self.krakensdr_source.set_freq(self.freq)
        self.qtgui_sink_ch0.set_frequency_range(self.freq, self.samp_rate)
        self.qtgui_sink_ch1.set_frequency_range(self.freq, self.samp_rate)


def main(top_block_cls=kraken_fft_2ch, options=None):
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
