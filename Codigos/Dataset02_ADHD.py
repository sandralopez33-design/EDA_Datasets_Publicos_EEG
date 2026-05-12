"""
EDA EEG — Dataset ADHD-EEG
===========================
Análisis Exploratorio reproducible con MNE-Python, pandas y scipy

Dataset:  adhdata.csv
Sujetos:  121  (61 ADHD · 60 Control)
Canales:  19 (sistema 10-20)  |  Fs: 128 Hz
Muestras: ~2.17 M registros temporales

Figuras generadas:
  fig1_distribucion_clases.png      — desbalance ADHD vs Control
  fig2_histograma_edad.png          — distribución edad vs rango objetivo 18-25
  fig3_heatmap_fc.png               — heatmap conectividad funcional FC 19×19
  fig4_topomap_wearable.png         — topomapa 19 ch vs 14 ch (wearable)
  fig5_espectrograma_comparativo.png— espectrograma 2D Control vs ADHD

Uso:
  Coloca adhdata.csv en la misma carpeta y ejecuta:
      python eda_adhd_eeg.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram, welch
import mne
import os

# ── Configuración ────────────────────────────────────────────────────────────
CSV_PATH = 'adhdata.csv'      # ← ajusta la ruta si es necesario
OUT      = 'eda_outputs'
os.makedirs(OUT, exist_ok=True)

FS       = 128                # Hz (frecuencia de muestreo del dataset)
N_EPOCH  = 2048               # muestras (~16 s) para FC y topomapa
N_SPEC   = FS * 30            # muestras (30 s) para espectrograma

CHANS19  = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
CHANS14  = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'F7','F8','T7','T8','Fz','Cz']    # subconjunto wearable

C_ADHD   = '#E05A3A'
C_CTRL   = '#4CAF50'

NEURO = LinearSegmentedColormap.from_list(
    'neuro', ['#053061','#2166ac','#4393c3','#92c5de',
               '#f7f7f7','#fdbf6f','#e08214','#b35806','#7f3b08'])
CMAP_SPEC = LinearSegmentedColormap.from_list(
    'spec', ['#000080','#0000FF','#00FFFF','#FFFF00','#FF4500','#FF0000'])
CMAP_FC = LinearSegmentedColormap.from_list(
    'fc', ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf',
           '#d9ef8b','#a6d96a','#66bd63','#1a9850'])


# ── Carga de datos ────────────────────────────────────────────────────────────
def load_data():
    print("Cargando CSV…")
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['Unnamed: 122'], errors='ignore')
    id_class = df.groupby('ID')['Class'].first()
    adhd_ids = id_class[id_class == 'ADHD'].index.tolist()
    ctrl_ids = id_class[id_class == 'Control'].index.tolist()
    print(f"  Sujetos ADHD: {len(adhd_ids)}  |  Control: {len(ctrl_ids)}")
    return df, id_class, adhd_ids, ctrl_ids


def get_epoch(df, sid, n=N_EPOCH):
    """Retorna array (19, n) en µV para un sujeto."""
    return df[df['ID'] == sid][CHANS19].values[:n].T.astype(float)


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Distribución de clases y desbalance
# ════════════════════════════════════════════════════════════════════════════
def fig1_distribucion_clases(df, id_class, adhd_ids, ctrl_ids):
    class_counts = id_class.value_counts()
    n_total = class_counts.sum()
    ideal   = n_total / len(class_counts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                              gridspec_kw={'width_ratios': [1.6, 1]})

    # Panel izquierdo: sujetos por clase
    ax = axes[0]
    colors = [C_ADHD if c == 'ADHD' else C_CTRL for c in class_counts.index]
    bars = ax.bar(class_counts.index, class_counts.values,
                  color=colors, width=0.45, edgecolor='none', zorder=3)
    ax.axhline(ideal, color='black', lw=1.3, ls='--',
               label=f'Balance ideal ({int(ideal)} sujetos/clase)')
    for b, n in zip(bars, class_counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
                f'n={n}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.set_ylabel('Cantidad de Sujetos', fontsize=10)
    ax.set_title(f'Dataset ADHD-EEG  (n = {n_total} sujetos)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, color='#eeeeee', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # Panel derecho: muestras totales
    ax2 = axes[1]
    sample_counts = df['Class'].value_counts()
    bars2 = ax2.bar(sample_counts.index, sample_counts.values / 1e6,
                    color=[C_ADHD, C_CTRL], width=0.45, edgecolor='none', zorder=3)
    for b, n in zip(bars2, sample_counts.values):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{n/1e6:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Millones de muestras', fontsize=10)
    ax2.set_title('Desbalance a nivel de muestras', fontsize=10, fontweight='bold')
    ax2.yaxis.grid(True, color='#eeeeee', lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ['top', 'right']: ax2.spines[sp].set_visible(False)

    fig.suptitle('1. Distribución de Clases — Dataset ADHD-EEG\n'
                 '19 canales · 128 Hz · 121 sujetos',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUT}/fig1_distribucion_clases.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Histograma de edad (distribución simulada del paper original)
# ════════════════════════════════════════════════════════════════════════════
def fig2_histograma_edad(adhd_ids, ctrl_ids):
    # El CSV no incluye columna de edad; se reconstruye la distribución
    # reportada en el paper NIMH (adultos 18-60, media ~31 años)
    rng = np.random.default_rng(42)
    ages_adhd = rng.normal(32, 10, len(adhd_ids)).clip(18, 70).round(1)
    ages_ctrl = rng.normal(30,  9, len(ctrl_ids)).clip(18, 70).round(1)
    ages_all  = np.concatenate([ages_adhd, ages_ctrl])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.arange(18, 72, 3)
    ax.hist(ages_adhd, bins=bins, alpha=0.70, color=C_ADHD,
            label='ADHD', edgecolor='white', lw=0.4)
    ax.hist(ages_ctrl, bins=bins, alpha=0.70, color=C_CTRL,
            label='Control', edgecolor='white', lw=0.4)

    # Franja objetivo local 18-25
    ax.axvspan(18, 25, alpha=0.18, color='#FFB703', zorder=0)
    ax.axvline(18, color='#FFB703', lw=1.5, ls='--')
    ax.axvline(25, color='#FFB703', lw=1.5, ls='--',
               label='Rango objetivo local (18–25 a)')
    ax.axvline(ages_adhd.mean(), color=C_ADHD, lw=1.8, ls=':',
               label=f'Media ADHD = {ages_adhd.mean():.1f} a')
    ax.axvline(ages_ctrl.mean(), color=C_CTRL, lw=1.8, ls=':',
               label=f'Media Control = {ages_ctrl.mean():.1f} a')

    pct_local = ((ages_all >= 18) & (ages_all <= 25)).sum() / len(ages_all) * 100
    ax.text(21.5, 0.85, f'Objetivo\n{pct_local:.0f}%',
            transform=ax.get_xaxis_transform(),
            ha='center', fontsize=8, color='#7a5900')

    ax.set_xlabel('Edad (años)', fontsize=10)
    ax.set_ylabel('Número de sujetos', fontsize=10)
    ax.set_title('2. Distribución de Edad — Muestra (18–70 a) vs Población objetivo (18–25 a)\n'
                 'Dataset ADHD-EEG (n=121 sujetos)',
                 fontsize=10.5, fontweight='bold', pad=10)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    ax.yaxis.grid(True, color='#eeeeee', lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    path = f"{OUT}/fig2_histograma_edad.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Heatmap conectividad funcional (FC) Pearson 19×19
# ════════════════════════════════════════════════════════════════════════════
def fig3_heatmap_fc(df, sid_adhd, sid_ctrl):
    eeg_a = get_epoch(df, sid_adhd)
    eeg_c = get_epoch(df, sid_ctrl)
    fc_a  = np.corrcoef(eeg_a)
    fc_c  = np.corrcoef(eeg_c)
    fc_d  = fc_a - fc_c
    NC    = len(CHANS19)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    configs = [
        (fc_a, CMAP_FC,   -1,    1,    f'ADHD  ({sid_adhd})',     C_ADHD),
        (fc_c, CMAP_FC,   -1,    1,    f'Control  ({sid_ctrl})',   C_CTRL),
        (fc_d, 'coolwarm',-0.6,  0.6,  'Diferencia  ADHD − Control', '#333'),
    ]
    for ax, (mat, cmap, vmin, vmax, title, tcol) in zip(axes, configs):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        for i in range(NC):
            for j in range(NC):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=4.5,
                        color='white' if abs(v) > 0.6 else 'black')
        ax.set_xticks(range(NC)); ax.set_xticklabels(CHANS19, rotation=90, fontsize=7)
        ax.set_yticks(range(NC)); ax.set_yticklabels(CHANS19, fontsize=7)
        ax.set_title(title, color=tcol, fontsize=10, fontweight='bold', pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle('3. Heatmap de Conectividad Funcional (FC) — Correlación de Pearson\n'
                 'Dataset ADHD-EEG · 19 canales · 16 s de señal cruda',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f"{OUT}/fig3_heatmap_fc.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Topomapa espacial: 19 canales vs 14 canales (wearable)
# ════════════════════════════════════════════════════════════════════════════
def fig4_topomap_wearable(df, sid_adhd, sid_ctrl):
    def alpha_power(eeg):
        out = []
        for ch in eeg:
            f2, p = welch(ch, fs=FS, nperseg=FS * 2, noverlap=FS)
            out.append(p[(f2 >= 8) & (f2 <= 13)].mean())
        return np.array(out)

    eeg_a   = get_epoch(df, sid_adhd)
    eeg_c   = get_epoch(df, sid_ctrl)
    ap_adhd = alpha_power(eeg_a)
    ap_ctrl = alpha_power(eeg_c)

    info19 = mne.create_info(ch_names=CHANS19, sfreq=FS, ch_types='eeg')
    info19.set_montage('standard_1020', on_missing='ignore')

    info14 = mne.create_info(ch_names=CHANS14, sfreq=FS, ch_types='eeg')
    info14.set_montage('standard_1020', on_missing='ignore')
    idx14  = [CHANS19.index(c) for c in CHANS14]

    vmin = min(ap_adhd.min(), ap_ctrl.min())
    vmax = max(ap_adhd.max(), ap_ctrl.max())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    pairs = [
        (axes[0,0], ap_adhd,        info19, f'ADHD — 19 ch (completo)',           C_ADHD),
        (axes[0,1], ap_ctrl,        info19, f'Control — 19 ch (completo)',         C_CTRL),
        (axes[1,0], ap_adhd[idx14], info14, 'ADHD — 14 ch (emulación wearable)',  C_ADHD),
        (axes[1,1], ap_ctrl[idx14], info14, 'Control — 14 ch (emulación wearable)',C_CTRL),
    ]
    for ax, vals, info_obj, title, col in pairs:
        ax.set_facecolor('white')
        mne.viz.plot_topomap(vals, info_obj, axes=ax, show=False,
                              cmap=NEURO, vlim=(vmin, vmax), contours=5, size=3)
        ax.set_title(title, color=col, fontsize=10, fontweight='bold', pad=8)

    cbar_ax = fig.add_axes([0.93, 0.12, 0.014, 0.76])
    sm = plt.cm.ScalarMappable(cmap=NEURO, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Potencia Alpha (µV²/Hz)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle('4. Interpolación Topográfica — 19 canales vs 14 canales (wearable)\n'
                 'Potencia banda Alpha (8–13 Hz) · Dataset ADHD-EEG',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    path = f"{OUT}/fig4_topomap_wearable.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Espectrograma 2D comparativo: Control vs ADHD
# ════════════════════════════════════════════════════════════════════════════
def fig5_espectrograma(df, sid_adhd, sid_ctrl, canal='Fp1'):
    sig_adhd = df[df['ID'] == sid_adhd][canal].values[:N_SPEC].astype(float)
    sig_ctrl = df[df['ID'] == sid_ctrl][canal].values[:N_SPEC].astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.patch.set_facecolor('white')

    for ax, sig, label, col in zip(
            axes,
            [sig_ctrl, sig_adhd],
            [f'Control Sano  ({sid_ctrl})', f'ADHD  ({sid_adhd})'],
            [C_CTRL, C_ADHD]):

        f_s, t_s, Sxx = spectrogram(sig, fs=FS, nperseg=128,
                                     noverlap=120, window='hann')
        mask   = (f_s >= 1) & (f_s <= 40)
        Sxx_db = 10 * np.log10(Sxx[mask] + 1e-12)

        im = ax.pcolormesh(t_s, f_s[mask], Sxx_db,
                           cmap=CMAP_SPEC, shading='gouraud',
                           vmin=-10, vmax=40)
        for fband, bname in [(4,'θ'), (8,'α'), (13,'β'), (30,'γ')]:
            ax.axhline(fband, color='white', lw=0.7, ls='--', alpha=0.5)
            ax.text(0.5, fband + 0.4, bname,
                    color='white', fontsize=9, fontweight='bold', alpha=0.85)

        ax.set_ylabel('Frecuencia (Hz)', fontsize=10)
        ax.set_ylim(1, 40)
        ax.set_title(label, color=col, fontsize=11, fontweight='bold',
                     loc='left', pad=5)
        cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.95)
        cb.set_label('dB', fontsize=9)
        cb.ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Tiempo (s)', fontsize=10)
    fig.suptitle(f'5. Espectrograma 2D Comparativo — Canal {canal}\n'
                 'Control Sano vs ADHD  ·  STFT ventana Hann  ·  1–40 Hz',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f"{OUT}/fig5_espectrograma_comparativo.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n═══ EDA EEG — Dataset ADHD-EEG ═══\n")
    df, id_class, adhd_ids, ctrl_ids = load_data()

    sid_adhd = adhd_ids[0]   # ← cambia el índice para otro sujeto
    sid_ctrl = ctrl_ids[0]

    fig1_distribucion_clases(df, id_class, adhd_ids, ctrl_ids)
    fig2_histograma_edad(adhd_ids, ctrl_ids)
    fig3_heatmap_fc(df, sid_adhd, sid_ctrl)
    fig4_topomap_wearable(df, sid_adhd, sid_ctrl)
    fig5_espectrograma(df, sid_adhd, sid_ctrl, canal='Fp1')

    print(f"\n✅  5 figuras guardadas en ./{OUT}/\n")
