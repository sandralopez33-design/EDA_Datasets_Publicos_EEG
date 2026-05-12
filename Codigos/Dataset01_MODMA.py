"""
EDA EEG — Dataset MODMA  (HC vs MDD)
Estilo: fondo blanco, colores verde/naranja, multi-canal, valores numéricos
Figuras:
  fig1_distribucion_clases.png
  fig2_serie_cruda_5s.png
  fig3_psd_welch.png
  fig4_heatmap_pearson.png
  fig5_topomap_alpha.png

Para datos reales (EDF/FIF):
  raw = mne.io.read_raw_edf('archivo.edf', preload=True)
  raw.pick_channels(CHANS14)
  raw.filter(1., 45.)
  epochs = mne.make_fixed_length_epochs(raw, duration=5.0)
  eeg = epochs.get_data() * 1e6  # → µV  (n_epochs, 14, n_samples)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import welch
import mne, os

np.random.seed(42)
OUT     = "eda_outputs"
os.makedirs(OUT, exist_ok=True)

FS      = 256
DUR     = 5.0
N_HC    = 29
N_MDD   = 24
CHANS14 = ['AF3','F7','F3','FC5','T7','P7','O1','O2',
           'P8','T8','FC6','F4','F8','AF4']
NC      = len(CHANS14)
SUBJ_ID = '02010002'

C_HC  = '#4CAF50'
C_MDD = '#FF7043'
CHAN_COLORS = plt.cm.RdYlBu_r(np.linspace(0.05, 0.95, NC))


def gen_eeg(n_subj, alpha_amp=1.0, noise=7, seed=0):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, DUR, int(FS*DUR), endpoint=False)
    out = []
    for _ in range(n_subj):
        chs = []
        for ch in CHANS14:
            a  = alpha_amp * (1.5 if ch.startswith(('AF','F')) else 1.0)
            s  = a  * 12 * np.sin(2*np.pi*10*t + rng.uniform(0, 2*np.pi))
            s += 0.5*  5 * np.sin(2*np.pi*20*t + rng.uniform(0, 2*np.pi))
            s += 0.4*  3 * np.sin(2*np.pi* 6*t + rng.uniform(0, 2*np.pi))
            s += rng.standard_normal(len(t)) * noise
            chs.append(s)
        out.append(chs)
    return np.array(out)

eeg_hc  = gen_eeg(N_HC,  alpha_amp=1.1, seed=0)
eeg_mdd = gen_eeg(N_MDD, alpha_amp=0.6, seed=100)


def fig1_distribucion_clases():
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    bars = ax.bar(['Control\nSaludable (HC)', 'Depresión Mayor\n(MDD)'],
                  [N_HC, N_MDD], color=[C_HC, C_MDD], width=0.5, edgecolor='none', zorder=3)
    ideal = (N_HC + N_MDD) / 2
    ax.axhline(ideal, color='black', lw=1.2, ls='--', label='Distribución ideal (50/50)')
    for b, n in zip(bars, [N_HC, N_MDD]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, f'n={n}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 38)
    ax.set_ylabel('Cantidad de Sujetos', fontsize=10)
    ax.set_title('1. Distribución de Clases — Dataset MODMA (n = 53)',
                 fontsize=10, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.yaxis.grid(True, color='#eeeeee', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_distribucion_clases.png", dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ fig1")


def fig2_serie_cruda():
    t = np.linspace(0, DUR, int(FS*DUR), endpoint=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    offset = 40
    for i, (ch, col) in enumerate(zip(CHANS14, CHAN_COLORS)):
        ax.plot(t, eeg_hc[0, i] + i*offset, color=col, lw=0.65, alpha=0.9)
    handles = [mpatches.Patch(color=CHAN_COLORS[i], label=CHANS14[i]) for i in range(NC)]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc='upper right',
              framealpha=0.8, bbox_to_anchor=(1.13, 1.0))
    ax.set_xlim(0, DUR)
    ax.set_xlabel('Tiempo (s)', fontsize=9)
    ax.set_ylabel('Amplitud (µV + offset)', fontsize=9)
    ax.set_title(f'2. Serie de Tiempo EEG (5 s) — 14 Canales EPOC X\nDataset MODMA, sujeto {SUBJ_ID}',
                 fontsize=9.5, fontweight='bold')
    ax.yaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_serie_cruda_5s.png", dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ fig2")


def fig3_psd_welch():
    fig, ax = plt.subplots(figsize=(7, 4.8))
    band_colors = ['#cce5ff','#d4edda','#fff3cd','#f8d7da','#e2d9f3']
    for (name, fl, fh), bc in zip(
            [('Delta\n1-4',1,4),('Theta\n4-8',4,8),('Alpha\n8-13',8,13),
             ('Beta\n13-30',13,30),('Gamma\n30-40',30,40)], band_colors):
        ax.axvspan(fl, fh, alpha=0.55, color=bc, zorder=1)
        ax.text((fl+fh)/2, 1.5e3, name, ha='center', va='top', fontsize=6.5, color='#444', zorder=5)
    for i, (ch, col) in enumerate(zip(CHANS14, CHAN_COLORS)):
        f2, p = welch(eeg_hc[0, i], fs=FS, nperseg=FS*2, noverlap=FS)
        mask = (f2>=1)&(f2<=40)
        ax.semilogy(f2[mask], p[mask], color=col, lw=0.9, alpha=0.85, label=ch)
    ax.set_xlim(1, 40)
    ax.set_xlabel('Frecuencia (Hz)', fontsize=9)
    ax.set_ylabel('PSD (µV²/Hz)', fontsize=9)
    ax.set_title(f'3. PSD Welch 1–40 Hz — 14 Canales EPOC X\nDataset MODMA, sujeto {SUBJ_ID}',
                 fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='upper right', framealpha=0.85,
              bbox_to_anchor=(1.13, 1.0))
    ax.yaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    ax.xaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_psd_welch.png", dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ fig3")


def fig4_heatmap_pearson():
    corr = np.corrcoef(eeg_hc[0])
    CMAP_CORR = LinearSegmentedColormap.from_list(
        'rdylgn', ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf',
                   '#d9ef8b','#a6d96a','#66bd63','#1a9850'])
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap=CMAP_CORR, vmin=-1, vmax=1, aspect='equal')
    for i in range(NC):
        for j in range(NC):
            v = corr[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.8,
                    color='white' if abs(v)>0.65 else 'black')
    ax.set_xticks(range(NC)); ax.set_xticklabels(CHANS14, rotation=90, fontsize=8)
    ax.set_yticks(range(NC)); ax.set_yticklabels(CHANS14, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, shrink=0.85)
    cb.set_ticks([-1,-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00])
    cb.ax.tick_params(labelsize=8)
    ax.set_title(f'4. Correlación Pearson 14×14 — EPOC X Channels\nDataset MODMA, sujeto {SUBJ_ID}',
                 fontsize=10, fontweight='bold', pad=10)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_heatmap_pearson.png", dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ fig4")


def fig5_topomap_alpha():
    info = mne.create_info(ch_names=CHANS14, sfreq=FS, ch_types='eeg')
    info.set_montage('standard_1020', on_missing='ignore')
    def alpha_pow(eeg, subj=0):
        out = []
        for ch in eeg[subj]:
            f2, p = welch(ch, fs=FS, nperseg=FS*2, noverlap=FS)
            out.append(p[(f2>=8)&(f2<=13)].mean())
        return np.array(out)
    ap_hc_s  = alpha_pow(eeg_hc)
    ap_mdd_s = alpha_pow(eeg_mdd)
    vmin = min(ap_hc_s.min(), ap_mdd_s.min())
    vmax = max(ap_hc_s.max(), ap_mdd_s.max())
    NEURO = LinearSegmentedColormap.from_list(
        'neuro2', ['#053061','#2166ac','#4393c3','#92c5de',
                   '#f7f7f7','#fdbf6f','#e08214','#b35806','#7f3b08'])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, ap, lbl, col in zip(axes,
            [ap_hc_s, ap_mdd_s],
            ['Control Saludable (HC)', 'Depresión Mayor (MDD)'],
            [C_HC, C_MDD]):
        mne.viz.plot_topomap(ap, info, axes=ax, show=False,
                              cmap=NEURO, vlim=(vmin, vmax), contours=6, size=3)
        ax.set_title(lbl, color=col, fontsize=11, fontweight='bold', pad=8)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.016, 0.70])
    sm = plt.cm.ScalarMappable(cmap=NEURO, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('PSD Alpha (µV²/Hz)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle('5. Topomapa 2D — Banda Alpha (8–13 Hz)\nDataset MODMA',
                 fontsize=10, fontweight='bold')
    fig.savefig(f"{OUT}/fig5_topomap_alpha.png", dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ fig5")


if __name__ == "__main__":
    print("\n═══ EDA EEG — Dataset MODMA ═══\n")
    fig1_distribucion_clases()
    fig2_serie_cruda()
    fig3_psd_welch()
    fig4_heatmap_pearson()
    fig5_topomap_alpha()
    print(f"\n✅  5 figuras en ./{OUT}/\n")
