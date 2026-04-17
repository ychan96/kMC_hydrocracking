import numpy as np
import time
import sys
from typing import Optional, Union
from collections import deque

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QProgressBar,
    QSpinBox, QDoubleSpinBox, QTextEdit, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

from kmc_v3.init import BaseKineticMC
from kmc_v3.count_sites import ConfigMixin
from kmc_v3.reactions import ReactionMixin
from kmc_v3.utils import identify_final_products


# ══════════════════════════════════════════════════════════════════
#  KMC class
# ══════════════════════════════════════════════════════════════════
class KMC(BaseKineticMC, ConfigMixin, ReactionMixin):
    pass


# ══════════════════════════════════════════════════════════════════
#  Dark colour tokens  (mirrors the HTML CSS variables)
# ══════════════════════════════════════════════════════════════════
D = {
    'bg':     '#0d1117',
    'panel':  '#161b22',
    'item':   '#21262d',
    'border': '#30363d',
    'txt':    '#c9d1d9',
    'muted':  '#8b949e',
    'blue':   '#79c0ff',
    'green':  '#3fb950',
    'warn':   '#d29922',
    'danger': '#f85149',
    'accent': '#1f6feb',
    # site colours
    'c_vacant': '#1c3a52',
    'c_smc':    '#d97742',
    'c_dmc':    '#d32f2f',
    'h_occ':    '#3cb371',
    'h_vac':    '#12291a',
}


# ══════════════════════════════════════════════════════════════════
#  SimWorker  — runs the KMC loop in a background QThread
#  Emits a state snapshot every `emit_every` successful steps so
#  the GUI can redraw without blocking the simulation.
# ══════════════════════════════════════════════════════════════════
class SimWorker(QObject):
    step_done = pyqtSignal(dict)     # live snapshot
    finished  = pyqtSignal(dict)     # final result
    log_msg   = pyqtSignal(str, str) # (text, level)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg      = cfg
        self._running = True
        self._paused  = False

    def stop(self):   self._running = False
    def pause(self):  self._paused  = True
    def resume(self): self._paused  = False

    def run(self):
        c   = self.cfg
        sim = KMC(
            temp_C         = c['temp_C'],
            reaction_time  = c['reaction_time'],
            chain_length   = c['chain_length'],
            params         = c.get('kmc_params'),
            P_H2           = c['P_H2'],
            catalyst_config= c.get('catalyst_config'),
        )
        coords    = sim.surface.get_coordinates_array()
        c_idx     = sim.surface.c_site_indices
        h_idx     = sim.surface.h_site_indices
        c_coords  = coords[c_idx]
        h_coords  = coords[h_idx]

        self.log_msg.emit(
            f'[init] Pt(111)  {sim.surface.n_c_sites} C / {sim.surface.n_h_sites} H sites', 'info')
        self.log_msg.emit(
            f'[init] chain={sim.chain_length}  T={c["temp_C"]}°C  P={c["P_H2"]} bar', 'info')
        self.log_msg.emit(
            f'[init] \u03b8_H seeded {sim.theta_H*100:.1f}%', 'ok')

        steps      = 0
        emit_every = c.get('emit_every', 20)
        max_steps  = c.get('max_steps')   # None = unlimited
        t0         = time.time()

        while sim.current_time < sim.reaction_time:
            if not self._running:
                break
            while self._paused:
                time.sleep(0.04)
                if not self._running:
                    break

            if max_steps and steps >= max_steps:
                break

            counts      = sim.update_configuration()
            key, dt     = sim.select_reaction(counts)
            if key is None:
                self.log_msg.emit('[halt] no available reactions', 'warn')
                break

            if sim.perform_reaction(key):
                sim.current_time += dt
                steps            += 1
                rtype, N, pos     = key

                if steps % emit_every == 0:
                    self.step_done.emit({
                        'step':      steps,
                        'sim_time':  sim.current_time,
                        'wall_time': time.time() - t0,
                        'reaction':  key,
                        'occ':       sim.occupancy.copy(),
                        'chain_at':  sim.chain_at_site.copy(),
                        'h_occ':     sim.h_occupancy.copy(),
                        'c_coords':  c_coords,
                        'h_coords':  h_coords,
                    })

                lvl = 'warn' if rtype == 'cracking' else \
                      'info' if rtype == 'dMC'      else 'ok'
                self.log_msg.emit(
                    f'[{steps}] {rtype}  N={N}  pos={pos}  '
                    f't={sim.current_time:.5f}s', lvl)

        elapsed  = time.time() - t0
        products = identify_final_products(sim.chain_array)
        self.log_msg.emit(
            f'[done] {steps} steps  sim={sim.current_time:.4f}s  '
            f'wall={elapsed:.2f}s', 'ok')
        self.finished.emit({
            'carbon_array':     sim.carbon_array.copy(),
            'chain_array':      sim.chain_array.copy(),
            'time':             sim.current_time,
            'products':         products,
            'steps':            steps,
            'computation_time': elapsed,
        })


# ══════════════════════════════════════════════════════════════════
#  SurfaceCanvas  — matplotlib figure embedded in Qt
# ══════════════════════════════════════════════════════════════════
class SurfaceCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=D['bg'])
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._clear()

    def _clear(self):
        self.ax.cla()
        self.ax.set_facecolor(D['bg'])
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.tight_layout(pad=0.4)

    def update_state(self, s: dict):
        occ      = s['occ']
        chain_at = s['chain_at']
        h_occ    = s['h_occ']
        c_coords = s['c_coords']
        h_coords = s['h_coords']

        self._clear()
        ax = self.ax

        # ── H hollow sites — small plain circles, no label ──
        if len(h_coords):
            mv = h_occ == 0
            mo = h_occ == 1
            if np.any(mv):
                ax.scatter(h_coords[mv, 0], h_coords[mv, 1],
                           s=15, c=D['h_vac'],
                           edgecolors='#2d5c3a', linewidths=0.4, zorder=2)
            if np.any(mo):
                ax.scatter(h_coords[mo, 0], h_coords[mo, 1],
                           s=15, c=D['h_occ'],
                           edgecolors='#256d3e', linewidths=0.4, zorder=2)

        # ── dMC bond lines ──
        dmc_idx = np.where(occ == 2)[0]
        drawn   = set()
        for k in dmc_idx:
            if k in drawn:
                continue
            for m in dmc_idx:
                if m == k or m in drawn:
                    continue
                if chain_at[m] == chain_at[k]:
                    d = np.hypot(c_coords[k, 0] - c_coords[m, 0],
                                 c_coords[k, 1] - c_coords[m, 1])
                    if d < 5.5:
                        ax.plot([c_coords[k, 0], c_coords[m, 0]],
                                [c_coords[k, 1], c_coords[m, 1]],
                                color='#a02020', lw=3, alpha=0.4,
                                solid_capstyle='round', zorder=3)
                        drawn.add(k); drawn.add(m)
                        break

        # ── C atop sites ──
        col_map = {0: D['c_vacant'], 1: D['c_smc'], 2: D['c_dmc']}
        for state in (0, 1, 2):
            mask = occ == state
            if not np.any(mask):
                continue
            ax.scatter(c_coords[mask, 0], c_coords[mask, 1],
                       s=110, c=col_map[state],
                       edgecolors='white', linewidths=0.7, zorder=4)

        # ── Fragment-length labels ──
        for i in range(len(occ)):
            if occ[i] != 0:
                ax.text(c_coords[i, 0], c_coords[i, 1],
                        str(abs(chain_at[i])),
                        ha='center', va='center',
                        fontsize=5, fontweight='bold',
                        color='white', zorder=5)

        # ── Title ──
        n_c   = len(occ)
        n_occ = int(np.sum(occ > 0))
        n_h   = int(np.sum(h_occ == 1))
        tc    = n_occ / n_c     * 100 if n_c          else 0.0
        th    = n_h   / len(h_occ) * 100 if len(h_occ) else 0.0
        ax.set_title(
            f'\u03b8_C: {tc:.1f}%    \u03b8_H: {th:.1f}%    '
            f't = {s["sim_time"]:.4f} s',
            color=D['muted'], fontsize=9, pad=6)

        self.draw()


# ══════════════════════════════════════════════════════════════════
#  Small widget helpers
# ══════════════════════════════════════════════════════════════════
def _lbl(text, color=None, size=10, bold=False):
    w = QLabel(text)
    w.setFont(QFont('Courier New', size,
                    QFont.Bold if bold else QFont.Normal))
    w.setStyleSheet(f'color:{color or D["muted"]};')
    return w

def _section(title):
    w = QLabel(f'  {title}')
    w.setFixedHeight(20)
    w.setFont(QFont('Courier New', 8))
    w.setStyleSheet(
        f'background:{D["item"]};color:{D["muted"]};'
        f'border-top:1px solid {D["border"]};'
        f'border-bottom:1px solid {D["border"]};'
        f'letter-spacing:1px;')
    return w

def _dspin(val, lo, hi, step):
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step)
    w.setFixedWidth(82)
    w.setFont(QFont('Courier New', 9))
    w.setStyleSheet(
        f'background:{D["item"]};color:{D["blue"]};'
        f'border:1px solid {D["border"]};border-radius:3px;padding:1px 4px;')
    return w

def _ispin(val, lo, hi, step):
    w = QSpinBox()
    w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step)
    w.setFixedWidth(82)
    w.setFont(QFont('Courier New', 9))
    w.setStyleSheet(
        f'background:{D["item"]};color:{D["blue"]};'
        f'border:1px solid {D["border"]};border-radius:3px;padding:1px 4px;')
    return w

def _pbar(accent):
    b = QProgressBar()
    b.setRange(0, 100); b.setValue(0)
    b.setFixedHeight(4); b.setTextVisible(False)
    b.setStyleSheet(
        f'QProgressBar{{background:{D["item"]};border:none;border-radius:2px;}}'
        f'QProgressBar::chunk{{background:{accent};border-radius:2px;}}')
    return b

def _btn(text, bg, border_col, slot):
    b = QPushButton(text)
    b.setFont(QFont('Courier New', 9))
    b.setFixedHeight(26)
    b.setStyleSheet(
        f'QPushButton{{background:{bg};color:{D["txt"]};'
        f'border:1px solid {border_col};border-radius:4px;padding:0 8px;}}'
        f'QPushButton:hover{{background:{D["bg"]};}}'
        f'QPushButton:disabled{{color:{D["muted"]};}}')
    b.clicked.connect(slot)
    return b

def _param_row(label_text, widget):
    row = QHBoxLayout()
    row.setContentsMargins(10, 2, 10, 2)
    row.addWidget(_lbl(label_text, size=9))
    row.addStretch()
    row.addWidget(widget)
    return row

def _info_row(label_text, value_text):
    row = QHBoxLayout()
    row.setContentsMargins(10, 2, 10, 2)
    row.addWidget(_lbl(label_text, size=9))
    row.addStretch()
    row.addWidget(_lbl(value_text, D['blue'], size=9))
    return row

def _stat_label(key):
    w = QLabel(f'{key}  —')
    w.setFont(QFont('Courier New', 9))
    w.setStyleSheet(f'color:{D["muted"]};')
    w._key = key
    return w

def _update_stat(w, val):
    w.setText(f'{w._key}  <span style="color:{D["green"]}">{val}</span>')

def _hline():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f'color:{D["border"]};')
    return f


# ══════════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════════
class KMCGui(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('KMC Surface Viewer — Pt(111)')
        self.resize(1300, 820)
        self._worker = None
        self._thread = None
        self._build()
        self.setStyleSheet(f'QMainWindow{{background:{D["bg"]};}}')

    # ── Build UI ───────────────────────────────────────────────────
    def _build(self):
        root = QWidget(); self.setCentralWidget(root)
        lay  = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        lay.addWidget(self._sidebar())
        lay.addWidget(self._centre(), stretch=1)
        lay.addWidget(self._rightpanel())

    # ── Left sidebar ───────────────────────────────────────────────
    def _sidebar(self):
        w = QWidget(); w.setFixedWidth(214)
        w.setStyleSheet(
            f'background:{D["panel"]};'
            f'border-right:1px solid {D["border"]};')
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        # System info
        lay.addWidget(_section('SYSTEM'))
        lay.addLayout(_info_row('Metal',       'Pt(111)'))
        lay.addLayout(_info_row('Lattice (Å)', '3.92'))
        lay.addLayout(_info_row('Grid',        '20 × 20'))
        lay.addLayout(_info_row('C sites',     '400'))
        lay.addLayout(_info_row('H sites',     '400'))

        # Editable parameters
        lay.addWidget(_section('PARAMETERS'))
        self.sp_temp  = _dspin(250,  100, 600,     1)
        self.sp_time  = _dspin(7200, 100, 500000, 100)
        self.sp_chain = _ispin(300,  10,  2000,   10)
        self.sp_P     = _dspin(50,   1,   300,     1)
        lay.addLayout(_param_row('T (°C)',       self.sp_temp))
        lay.addLayout(_param_row('Time (s)',     self.sp_time))
        lay.addLayout(_param_row('Chain length', self.sp_chain))
        lay.addLayout(_param_row('P_H₂ (bar)',   self.sp_P))

        # Coverage bars
        lay.addWidget(_section('COVERAGE'))
        self.lbl_tc = _lbl('θ_C   0.0%', D['blue'],  size=9)
        self.bar_tc = _pbar(D['blue'])
        self.lbl_th = _lbl('θ_H  38.0%', D['green'], size=9)
        self.bar_th = _pbar(D['green'])
        for w2 in [self.lbl_tc, self.bar_tc, self.lbl_th, self.bar_th]:
            pw = QWidget(); pl = QVBoxLayout(pw)
            pl.setContentsMargins(10, 2, 10, 2); pl.setSpacing(1)
            pl.addWidget(w2); lay.addWidget(pw)

        # Controls
        lay.addWidget(_section('CONTROLS'))
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(8, 6, 8, 6); btn_row.setSpacing(5)
        self.btn_run   = _btn('▶  Run',   D['accent'], D['accent'], self._run)
        self.btn_pause = _btn('⏸  Pause', D['item'],   D['border'], self._pause)
        self.btn_reset = _btn('↺  Reset', D['item'],   D['border'], self._reset)
        self.btn_pause.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_reset)
        lay.addLayout(btn_row)

        # Sim options
        lay.addWidget(_section('OPTIONS'))
        self.sp_emit    = _ispin(20,  1, 1000,  10)
        self.sp_maxstep = _ispin(0,   0, 10000000, 1000)
        lay.addLayout(_param_row('Emit every N',   self.sp_emit))
        lay.addLayout(_param_row('Max steps (0=∞)', self.sp_maxstep))

        lay.addStretch()
        return w

    # ── Centre (canvas + toolbar + statusbar) ─────────────────────
    def _centre(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        # Toolbar
        tb = QWidget(); tb.setFixedHeight(34)
        tb.setStyleSheet(
            f'background:{D["panel"]};'
            f'border-bottom:1px solid {D["border"]};')
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(10, 0, 10, 0); tb_lay.setSpacing(8)
        tb_lay.addWidget(_lbl('KMC Surface Viewer — live simulation', D['muted'], size=9))
        tb_lay.addStretch()
        lay.addWidget(tb)

        # Canvas
        self.canvas = SurfaceCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.canvas, stretch=1)

        # Status bar
        sb = QWidget(); sb.setFixedHeight(26)
        sb.setStyleSheet(
            f'background:{D["panel"]};'
            f'border-top:1px solid {D["border"]};')
        sb_lay = QHBoxLayout(sb)
        sb_lay.setContentsMargins(10, 0, 10, 0); sb_lay.setSpacing(20)
        self.st_step  = _stat_label('step')
        self.st_time  = _stat_label('sim time')
        self.st_cocc  = _stat_label('C occ')
        self.st_hocc  = _stat_label('H occ')
        self.st_last  = _stat_label('last rxn')
        for s in [self.st_step, self.st_time, self.st_cocc,
                  self.st_hocc, self.st_last]:
            sb_lay.addWidget(s)
        sb_lay.addStretch()
        lay.addWidget(sb)
        return w

    # ── Right panel (legend + log) ─────────────────────────────────
    def _rightpanel(self):
        w = QWidget(); w.setFixedWidth(194)
        w.setStyleSheet(
            f'background:{D["panel"]};'
            f'border-left:1px solid {D["border"]};')
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        lay.addWidget(_section('LEGEND'))
        for col, label in [
            (D['c_vacant'], 'Vacant C (atop)'),
            (D['c_smc'],    'Single M-C'),
            (D['c_dmc'],    'dMC'),
            (D['h_occ'],    'H hollow (occ.)'),
            (D['h_vac'],    'H hollow (vacant)'),
        ]:
            row = QHBoxLayout(); row.setContentsMargins(10, 3, 10, 3)
            dot = QLabel('●'); dot.setFont(QFont('Courier New', 12))
            dot.setStyleSheet(f'color:{col};')
            row.addWidget(dot)
            row.addWidget(_lbl(label, size=9))
            row.addStretch()
            lay.addLayout(row)

        lay.addWidget(_section('REACTION LOG'))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            f'background:{D["bg"]};color:{D["muted"]};'
            f'font-family:Courier New;font-size:9px;'
            f'border:none;padding:4px;')
        lay.addWidget(self.log, stretch=1)
        return w

    # ── Simulation control ─────────────────────────────────────────
    def _run(self):
        if self._thread and self._thread.isRunning():
            return

        max_s = self.sp_maxstep.value() or None
        cfg   = {
            'temp_C':        self.sp_temp.value(),
            'reaction_time': self.sp_time.value(),
            'chain_length':  self.sp_chain.value(),
            'P_H2':          self.sp_P.value(),
            'emit_every':    self.sp_emit.value(),
            'max_steps':     max_s,
        }

        self._worker = SimWorker(cfg)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.step_done.connect(self._on_step)
        self._worker.finished.connect(self._on_finished)
        self._worker.log_msg.connect(self._on_log)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()
        self.btn_run.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_reset.setEnabled(False)
        self._on_log('[run] simulation started', 'info')

    def _pause(self):
        if not self._worker:
            return
        if '⏸' in self.btn_pause.text():
            self._worker.pause()
            self.btn_pause.setText('▶  Resume')
            self._on_log('[paused]', 'warn')
        else:
            self._worker.resume()
            self.btn_pause.setText('⏸  Pause')
            self._on_log('[resumed]', 'info')

    def _reset(self):
        if self._worker:
            self._worker.stop()
        if self._thread:
            self._thread.quit(); self._thread.wait(2000)
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.btn_pause.setText('⏸  Pause')
        self.log.clear()
        self._on_log('[reset] ready', 'info')

    # ── Live callbacks (called on main thread via signal) ──────────
    def _on_step(self, s: dict):
        # Redraw surface
        self.canvas.update_state(s)

        # Coverage bars + labels
        occ   = s['occ']
        h_occ = s['h_occ']
        n_occ = int(np.sum(occ > 0))
        n_h   = int(np.sum(h_occ == 1))
        tc    = n_occ / len(occ)     * 100 if len(occ)   else 0.0
        th    = n_h   / len(h_occ)   * 100 if len(h_occ) else 0.0
        self.lbl_tc.setText(f'\u03b8_C  {tc:.1f}%')
        self.lbl_th.setText(f'\u03b8_H  {th:.1f}%')
        self.bar_tc.setValue(int(tc))
        self.bar_th.setValue(int(th))

        # Status bar
        rtype, N, pos = s['reaction']
        _update_stat(self.st_step,  str(s['step']))
        _update_stat(self.st_time,  f'{s["sim_time"]:.4f} s')
        _update_stat(self.st_cocc,  str(n_occ))
        _update_stat(self.st_hocc,  str(n_h))
        _update_stat(self.st_last,  f'{rtype} N={N}')

    def _on_finished(self, result: dict):
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.btn_pause.setText('⏸  Pause')

    def _on_log(self, msg: str, level: str = ''):
        col = {
            'ok':   D['green'],
            'info': D['blue'],
            'warn': D['warn'],
            'err':  D['danger'],
        }.get(level, D['muted'])
        self.log.append(
            f'<span style="color:{col};font-family:Courier New;'
            f'font-size:9px">{msg}</span>')
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self._worker: self._worker.stop()
        if self._thread: self._thread.quit(); self._thread.wait(2000)
        event.accept()


# ══════════════════════════════════════════════════════════════════
#  Headless run_simulation  (unchanged public API)
# ══════════════════════════════════════════════════════════════════
def run_simulation(
    temp_C:         float                            = 250,
    reaction_time:  float                            = 7200,
    chain_length:   Optional[Union[int, np.ndarray]] = None,
    params:         Optional[dict]                   = None,
    P_H2:           float                            = 50,
    catalyst_config                                  = None,
    verbose:        bool                             = False,
    max_steps:      Optional[int]                    = None,
) -> dict:

    sim = KMC(
        temp_C=temp_C, 
        reaction_time=reaction_time,
        chain_length=chain_length, 
        params=params,
        P_H2=P_H2, 
        catalyst_config=catalyst_config,
    )

    history         = [] if verbose else None
    steps_performed = 0
    start_time      = time.time()

    #stagnation detection: if we perform too many non-cracking steps in a row, it's likely we've reached a dead end and should break to avoid infinite loops
    max_nonproductive = 1000
    non_productive    = 0

    while sim.current_time < sim.reaction_time and (max_steps is None or steps_performed < max_steps):
        # Count available sites
        counts  = sim.update_configuration()
        # Select reaction
        key, dt = sim.select_reaction(counts)

        if key is None:
            break

        # Perform reaction and update time
        if sim.perform_reaction(key):
            sim.current_time += dt
            steps_performed  += 1
            rtype, N, pos = key
            # verbose -> logging + history
            if verbose:
                products = identify_final_products(sim.chain_array)
                if history is not None:
                    history.append({
                        'step':         steps_performed,
                        'time':         sim.current_time,
                        'reaction':     key,
                        'carbon_array': sim.carbon_array.copy(),
                        'chain_array':  sim.chain_array.copy(),
                    })
                print(f'Step{steps_performed:>6d} / t={sim.current_time:.4f}s / 'f'({rtype}, N={N}, pos={pos})')
                print(f"Reaction: {rtype} / N = {N} / pos = {pos}")
                print(f"Carbon array: {sim.carbon_array}")
                print(f"Chain array: {sim.chain_array}")
                print(f"Hydrogen array: {sim.hydrogen_array}")
                print(f"Products after reaction: {products}")
                print("-" * 50)
                input('Press Enter to continue...')

            if rtype == 'cracking':
                non_productive = 0
            else:
                non_productive += 1

            if non_productive >= max_nonproductive:
                print(f'[warn] stagnation detected at step {steps_performed} — breaking')
                break

    elapsed  = time.time() - start_time
    products = identify_final_products(sim.chain_array)

    if verbose:
        print(f'\nDone  {elapsed:.2f}s  steps={steps_performed}')
        print(f"Final time: {sim.current_time:.2f}s, Steps: {steps_performed}")
        print(f"Final products: {products}")

    return {
        'carbon_array':     sim.carbon_array.copy(),
        'chain_array':      sim.chain_array.copy(),
        'time':             sim.current_time,
        'history':          history,
        'products':         products,
        'steps':            steps_performed,
        'computation_time': elapsed,
    }


# ══════════════════════════════════════════════════════════════════
#  Headless run_multiple_simulations
# ══════════════════════════════════════════════════════════════════
def run_multiple_simulations(
    num_sims:       int,
    temp_C:         float                            = 250,
    reaction_time:  float                            = 7200,
    chain_length:   Optional[Union[int, np.ndarray]] = None,
    params:         Optional[dict]                   = None,
    P_H2:           float                            = 50,
    catalyst_config                                  = None,
    verbose:        bool                             = False,
    max_steps:      Optional[int]                    = None,
    min_products:   Optional[int]                    = None,
    max_products:   Optional[int]                    = None,
) -> list:

    results     = []
    total_start = time.time()
    print(f'Running {num_sims} simulations at {temp_C}°C ...')

    for i in range(num_sims):
        result = run_simulation(
            temp_C=temp_C, 
            reaction_time=reaction_time,
            chain_length=chain_length, 
            params=params,
            P_H2=P_H2, 
            catalyst_config=catalyst_config,
            verbose=verbose, 
            max_steps=max_steps,
        )

        n = len(result['products'])

        print(f'[{i+1}/{num_sims}]: {len(result["carbon_array"])} carbon chains '
              f'sim={result["time"]:.4f}s  '
              f'products={n} in {result["steps"]} steps '
              f'({result["computation_time"]:.2f}s wall)')
        
        # ── per-simulation gate ───────────────────────────────────
        if min_products is not None and n < min_products:
            print(f'  [PRUNED] products={n} < min={min_products} — aborting batch')
            return None
 
        if max_products is not None and n > max_products:
            print(f'  [PRUNED] products={n} > max={max_products} — aborting batch')
            return None
        # ─────────────────────────────────────────────────────────
 
        results.append(result)

    print(f'\nAll {num_sims} done in {time.time()-total_start:.2f}s')

    return results


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════
def launch_gui():
    """Open the live KMC surface viewer window."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = KMCGui()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_gui()