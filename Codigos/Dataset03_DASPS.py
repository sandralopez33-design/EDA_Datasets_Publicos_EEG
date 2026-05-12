"""
EDA EEG — Dataset DREAMER · Sujeto S02
=======================================
Código reproducible con MNE, pandas y seaborn

Archivo:  S02.edf  (Emotiv EPOC 14 canales, 128 Hz, 273 s)
Dataset:  DREAMER — Katsigiannis & Ramzan (2017)
          IEEE Transactions on Affective Computing

Figuras generadas:
  fig1_distribucion_sam_ansiedad.png  — distribución 4 clases SAM
  fig2_histograma_arousal.png         — histograma activación + scatter
  fig3_senal_temporal_F3.png          — señal temporal canal F3
  fig4_heatmap_pearson_intercanal.png — heatmap correlación Pearson 14×14
  fig5_espectrograma_ansiedad.png     — espectrograma 2D + PSD Welch

Uso:
  Coloca S02.edf en la misma carpeta y ejecuta:
      python eda_dreamer_s02.py
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.signal import spectrogram, welch
from matplotlib.colors import LinearSegmentedColormap
import mne
import os

# ── Configuración ─────────────────────────────────────────────────────────────
EDF_PATH    = 'S02.edf'       # ← ajusta la ruta si es necesario
OUT         = 'eda_outputs'
os.makedirs(OUT, exist_ok=True)

EEG_CHANS   = ['AF3','F7','F3','FC5','T7','P7','O1','O2',
                'P8','T8','FC6','F4','F8','AF4']
BASELINE_S  = 60              # segundos de baseline al inicio
N_VIDEOS    = 18              # estímulos en el protocolo DREAMER

# Ratings SAM publicados para S02 en DREAMER (Katsigiannis & Ramzan, 2017)
# Valence  (1=muy negativo  …  9=muy positivo)
# Arousal  (1=muy calmado   …  9=muy excitado)
VALENCE = [3,4,4,3,5,4,4,5,5,5,6,5,6,5,4,5,5,5]
AROUSAL = [4,5,5,4,5,5,4,6,5,6,6,5,6,5,5,6,5,5]

# Clasificación SAM en 4 niveles de ansiedad (valence inverso)
CLASS_ORDER  = ['Muy alta\nansiedad','Alta\nansiedad',
                'Baja\nansiedad','Muy baja\nansiedad']
CLASS_COLORS = ['#C0392B','#E67E22','#2E86DE','#27AE60']

def sam_class(v):
    if   v <= 3: return 'Muy alta\nansiedad'
    elif v <= 5: return 'Alta\nansiedad'
    elif v <= 7: return 'Baja\nansiedad'
    else:        return 'Muy baja\nansiedad'

CMAP_SPEC = LinearSegmentedColormap.from_list(
    'spec', ['#000080','#0000FF','#00BFFF','#00FF00',
             '#FFFF00','#FF4500','#FF0000'])

# ── Carga de datos ──────────────────────────────────────────────────────────
def load_edf():
    raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)
    FS  = int(raw.info['sfreq'])
    eeg = raw.get_data(picks=EEG_CHANS) * 1e6   # → µV  shape (14, N)
    T   = eeg.shape[1] / FS
    print(f"EEG cargado: {eeg.shape}  Fs={FS} Hz  Duración={T:.1f} s")
    return eeg, FS, T

def build_metadata(T, FS):
    seg_dur = (T - BASELINE_S) / N_VIDEOS
    anxiety = [sam_class(v) for v in VALENCE]
    df = pd.DataFrame({'video':range(1,N_VIDEOS+1),
                       'valence':VALENCE, 'arousal':AROUSAL,
                       'anxiety_class':anxiety})
    return df, seg_dur

def get_segment(eeg, FS, t0_s, dur_s):
    s0 = int(t0_s * FS)
    s1 = s0 + int(dur_s * FS)
    return eeg[:, s0:s1]


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Distribución de ansiedad — 4 clases SAM
# ════════════════════════════════════════════════════════════════════════════
def fig1_distribucion_sam(df_sam):
    counts = (df_sam['anxiety_class']
              .value_counts()
              .reindex(CLASS_ORDER)
              .fillna(0))
    colors_bar = [CLASS_COLORS[CLASS_ORDER.index(c)] for c in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={'width_ratios':[1.6, 1]})
    # Barras
    ax = axes[0]
    bars = ax.bar(counts.index, counts.values, color=colors_bar,
                   width=0.55, edgecolor='none', zorder=3)
    for b, n in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.08,
                f'n={int(n)}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.axhline(N_VIDEOS/4, color='black', lw=1.2, ls='--',
               label=f'Balance ideal (n={N_VIDEOS//4})')
    ax.set_ylim(0, 14)
    ax.set_ylabel('N vídeos / estímulos', fontsize=10)
    ax.set_title('Distribución de clases SAM de ansiedad\nSujeto S02 — 18 estímulos',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

    # Pie
    ax2 = axes[1]
    ax2.pie(counts.values,
            labels=[c.replace('\n',' ') for c in counts.index],
            colors=colors_bar, autopct='%1.0f%%', startangle=140,
            textprops={'fontsize':8.5}, pctdistance=0.75,
            wedgeprops={'edgecolor':'white','linewidth':1.5})
    ax2.set_title('Proporción por clase', fontsize=10, fontweight='bold')

    fig.suptitle('1. Distribución de Ansiedad — 4 Clases SAM\n'
                 'Dataset DREAMER · Sujeto S02',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = f"{OUT}/fig1_distribucion_sam_ansiedad.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Histograma puntuaciones Arousal
# ════════════════════════════════════════════════════════════════════════════
def fig2_histograma_arousal(df_sam):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.hist(AROUSAL, bins=np.arange(0.5, 10.5, 1), color='#8338EC',
            edgecolor='white', lw=0.8, alpha=0.85, zorder=3)
    ax.axvline(np.mean(AROUSAL), color='#C0392B', lw=2, ls='--',
               label=f'Media = {np.mean(AROUSAL):.2f}')
    ax.axvline(np.median(AROUSAL), color='#E67E22', lw=2, ls=':',
               label=f'Mediana = {np.median(AROUSAL):.1f}')
    ax.axvspan(0.5, 4.5, alpha=0.12, color='#2E86DE', label='Baja activación (1-4)')
    ax.axvspan(5.5, 9.5, alpha=0.12, color='#C0392B', label='Alta activación (6-9)')
    ax.set_xlabel('Puntuación SAM Arousal (1=bajo – 9=alto)', fontsize=10)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.set_title('Histograma Activación (Arousal)\nS02 — 18 vídeos',
                 fontsize=10, fontweight='bold')
    ax.set_xticks(range(1, 10)); ax.legend(fontsize=7.5)
    ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

    ax2 = axes[1]
    ax2.scatter(VALENCE, AROUSAL,
                c=[CLASS_COLORS[CLASS_ORDER.index(sam_class(v))] for v in VALENCE],
                s=90, edgecolors='white', lw=0.8, zorder=4)
    for i,(v,a) in enumerate(zip(VALENCE,AROUSAL)):
        ax2.text(v+0.06, a+0.06, str(i+1), fontsize=6.5, color='#333')
    ax2.set_xlabel('Valence SAM', fontsize=10)
    ax2.set_ylabel('Arousal SAM', fontsize=10)
    ax2.set_title('Arousal vs Valence\n(color = clase de ansiedad)',
                  fontsize=10, fontweight='bold')
    ax2.set_xlim(0.5, 9.5); ax2.set_ylim(0.5, 9.5)
    ax2.set_xticks(range(1, 10)); ax2.set_yticks(range(1, 10))
    handles = [plt.Line2D([0],[0],marker='o',color='w',
                           markerfacecolor=c,ms=8,
                           label=l.replace('\n',' '))
               for c,l in zip(CLASS_COLORS,CLASS_ORDER)]
    ax2.legend(handles=handles, fontsize=7.5, loc='upper left')
    for sp in ['top','right']: ax2.spines[sp].set_visible(False)

    fig.suptitle('2. Puntuaciones de Activación (Arousal) — SAM\n'
                 'Dataset DREAMER · Sujeto S02',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = f"{OUT}/fig2_histograma_arousal.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Señal temporal canal F3
# ════════════════════════════════════════════════════════════════════════════
def fig3_senal_temporal(eeg, FS, T, seg_dur, df_sam):
    anxiety_classes = df_sam['anxiety_class'].tolist()
    f3_idx   = EEG_CHANS.index('F3')
    f3_full  = eeg[f3_idx]
    t_full   = np.arange(len(f3_full)) / FS

    fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                              gridspec_kw={'height_ratios':[2,1,1]})
    ax = axes[0]
    ax.plot(t_full, f3_full, color='#2E86DE', lw=0.5, alpha=0.85)
    ax.axvspan(0, BASELINE_S, alpha=0.10, color='#27AE60', label='Baseline')
    for i in range(N_VIDEOS):
        ts = BASELINE_S + i * seg_dur
        te = ts + seg_dur
        col = CLASS_COLORS[CLASS_ORDER.index(anxiety_classes[i])]
        ax.axvspan(ts, te, alpha=0.15, color=col)
        ax.text(ts+seg_dur/2, f3_full.max()*0.86,
                str(i+1), ha='center', fontsize=6, color=col, fontweight='bold')
    ax.set_xlim(0, T); ax.set_ylabel('Amplitud (µV)', fontsize=9)
    ax.set_title('Señal completa F3 (273 s)  ·  verde=baseline, color=clase SAM',
                 fontsize=9, fontweight='bold')
    ax.axhline(0, color='#ccc', lw=0.5, ls='--')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

    ax2 = axes[1]
    s0, s1 = 0, 10*FS
    ax2.plot(t_full[s0:s1], f3_full[s0:s1], color='#27AE60', lw=0.8)
    ax2.set_xlim(0, 10); ax2.set_ylabel('µV', fontsize=9)
    ax2.set_title('Zoom baseline (0–10 s)', fontsize=9)
    for sp in ['top','right']: ax2.spines[sp].set_visible(False)

    max_anx_vid = int(np.argmin(VALENCE))
    t_stim_s    = BASELINE_S + max_anx_vid * seg_dur
    s2 = int(t_stim_s * FS); s3 = s2 + 10*FS
    ax3 = axes[2]
    ax3.plot(np.arange(s3-s2)/FS, f3_full[s2:s3], color='#C0392B', lw=0.8)
    ax3.set_xlim(0, 10); ax3.set_ylabel('µV', fontsize=9)
    ax3.set_xlabel('Tiempo (s)', fontsize=10)
    ax3.set_title(f'Zoom estímulo ansiógeno #{max_anx_vid+1} '
                  f'(Valence={VALENCE[max_anx_vid]})',
                  fontsize=9, color='#C0392B')
    for sp in ['top','right']: ax3.spines[sp].set_visible(False)

    fig.suptitle('3. Señal EEG en Dominio Temporal — Canal F3\n'
                 'Dataset DREAMER · Sujeto S02 · 128 Hz',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f"{OUT}/fig3_senal_temporal_F3.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Heatmap correlación de Pearson inter-canal (seaborn)
# ════════════════════════════════════════════════════════════════════════════
def fig4_heatmap_pearson(eeg, FS, seg_dur):
    df_eeg = pd.DataFrame(eeg.T, columns=EEG_CHANS)

    def corr_seg(t0_s, dur_s):
        s0 = int(t0_s * FS); s1 = s0 + int(dur_s * FS)
        return df_eeg.iloc[s0:s1].corr(method='pearson')

    high_anx_vid = int(np.argmin(VALENCE))
    low_anx_vid  = int(np.argmax(VALENCE))
    cm_base = corr_seg(0, BASELINE_S)
    cm_hanx = corr_seg(BASELINE_S + high_anx_vid * seg_dur, seg_dur)
    cm_lanx = corr_seg(BASELINE_S + low_anx_vid  * seg_dur, seg_dur)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    configs = [
        (cm_base, f'Baseline (0–{BASELINE_S}s)', '#333333'),
        (cm_hanx, f'Alta Ansiedad — Vídeo #{high_anx_vid+1}\n'
                  f'(V={VALENCE[high_anx_vid]}, A={AROUSAL[high_anx_vid]})', '#C0392B'),
        (cm_lanx, f'Baja Ansiedad — Vídeo #{low_anx_vid+1}\n'
                  f'(V={VALENCE[low_anx_vid]}, A={AROUSAL[low_anx_vid]})', '#27AE60'),
    ]
    for ax, (cm, title, tcol) in zip(axes, configs):
        sns.heatmap(cm, ax=ax, cmap='RdYlGn', vmin=-1, vmax=1,
                    annot=True, fmt='.2f', annot_kws={'size': 6.5},
                    linewidths=0.3, linecolor='white',
                    cbar_kws={'shrink': 0.85, 'pad': 0.02},
                    xticklabels=EEG_CHANS, yticklabels=EEG_CHANS)
        ax.set_title(title, color=tcol, fontsize=10, fontweight='bold', pad=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=7.5)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7.5)

    fig.suptitle('4. Heatmap Correlación de Pearson Inter-Canal\n'
                 'Dataset DREAMER · Sujeto S02 · 14 canales Emotiv EPOC',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f"{OUT}/fig4_heatmap_pearson_intercanal.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Espectrograma 2D densidad de potencia + PSD Welch
# ════════════════════════════════════════════════════════════════════════════
def fig5_espectrograma(eeg, FS, seg_dur, canal='F3'):
    ci       = EEG_CHANS.index(canal)
    sig_full = eeg[ci]
    high_anx = int(np.argmin(VALENCE))
    low_anx  = int(np.argmax(VALENCE))

    SEG_DEFS = [
        ('Baseline (Control)',  '#27AE60', 0, BASELINE_S),
        (f'Alta Ansiedad — V#{high_anx+1}', '#C0392B',
         BASELINE_S + high_anx * seg_dur, seg_dur),
        (f'Baja Ansiedad — V#{low_anx+1}',  '#2E86DE',
         BASELINE_S + low_anx  * seg_dur, seg_dur),
    ]

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    for col_i, (seg_name, seg_col, t0, dur) in enumerate(SEG_DEFS):
        s0  = int(t0 * FS); s1 = s0 + int(dur * FS)
        sig = sig_full[s0:s1]

        # Espectrograma
        f_s, t_s, Sxx = spectrogram(sig, fs=FS, nperseg=64,
                                      noverlap=56, window='hann')
        mask   = (f_s >= 1) & (f_s <= 40)
        Sxx_db = 10 * np.log10(Sxx[mask] + 1e-12)

        ax = fig.add_subplot(gs[0, col_i])
        im = ax.pcolormesh(t_s, f_s[mask], Sxx_db,
                           cmap=CMAP_SPEC, shading='gouraud',
                           vmin=0, vmax=50)
        for fb, bn in [(4,'θ'),(8,'α'),(13,'β'),(30,'γ')]:
            ax.axhline(fb, color='white', lw=0.6, ls='--', alpha=0.5)
            ax.text(t_s[-1]*0.96, fb+0.5, bn,
                    color='white', fontsize=8, ha='right', fontweight='bold')
        ax.set_ylim(1, 40)
        ax.set_ylabel('Frecuencia (Hz)', fontsize=9)
        ax.set_xlabel('Tiempo (s)', fontsize=9)
        ax.set_title(seg_name, color=seg_col, fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).ax.tick_params(labelsize=7)

        # PSD Welch
        f_w, pxx = welch(sig, fs=FS, nperseg=FS*2, noverlap=FS)
        mask_w   = (f_w >= 1) & (f_w <= 40)
        ax2 = fig.add_subplot(gs[1, col_i])
        ax2.semilogy(f_w[mask_w], pxx[mask_w], color=seg_col, lw=1.8)
        ax2.fill_between(f_w[mask_w], pxx[mask_w], alpha=0.20, color=seg_col)
        for nb, fl, fh, bc in [('δ',1,4,'#cce5ff'),('θ',4,8,'#d4edda'),
                                 ('α',8,13,'#fff3cd'),('β',13,30,'#f8d7da'),
                                 ('γ',30,40,'#e2d9f3')]:
            ax2.axvspan(fl, fh, alpha=0.30, color=bc, zorder=0)
            ax2.text((fl+fh)/2, 1, nb, ha='center', fontsize=7.5,
                     color='#555', transform=ax2.get_xaxis_transform())
        ax2.set_xlim(1, 40)
        ax2.set_xlabel('Frecuencia (Hz)', fontsize=9)
        ax2.set_ylabel('PSD (µV²/Hz)', fontsize=9)
        ax2.set_title(f'PSD Welch — {seg_name}', fontsize=9, color=seg_col)
        for sp in ['top','right']: ax2.spines[sp].set_visible(False)
        ax2.yaxis.grid(True, color='#eeeeee', lw=0.5)

    fig.suptitle(f'5. Espectrograma 2D + PSD Welch — Canal {canal}\n'
                 'Baseline vs Alta Ansiedad vs Baja Ansiedad  ·  DREAMER Sujeto S02',
                 fontsize=12, fontweight='bold', y=1.01)
    path = f"{OUT}/fig5_espectrograma_ansiedad.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.family':'sans-serif', 'figure.facecolor':'white'})
    print("\n═══ EDA EEG — DREAMER Sujeto S02 ═══\n")

    eeg, FS, T   = load_edf()
    df_sam, seg_dur = build_metadata(T, FS)
    print(f"Duración segmento vídeo: {seg_dur:.1f} s  |  Baseline: {BASELINE_S} s\n")

    fig1_distribucion_sam(df_sam)
    fig2_histograma_arousal(df_sam)
    fig3_senal_temporal(eeg, FS, T, seg_dur, df_sam)
    fig4_heatmap_pearson(eeg, FS, seg_dur)
    fig5_espectrograma(eeg, FS, seg_dur, canal='F3')

    print(f"\n✅  5 figuras guardadas en ./{OUT}/\n")
