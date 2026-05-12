"""
EDA EEG — Dataset BRMH (Brain Mental Health)
=============================================
Rutinas reproducibles — datos reales del CSV

Dataset:  EEG.machinelearing_data_BRMH.csv
Sujetos:  945  |  Trastornos: 7  |  Columnas: 1149
Features: AB (potencia por banda, 19 ch × 6 bandas = 114 cols)
          COH (coherencia entre pares, 19ch × 6 bandas = 1026 cols)

Figuras generadas:
  fig1_distribucion_clases.png  — distribución de 7 clases
  fig2_psd_por_banda.png        — potencia AB por banda y trastorno
  fig3_perfil_espectral.png     — perfil espectral por trastorno
  fig4_heatmap_pearson.png      — heatmap coherencia alpha 14×14
  fig5_topomap_alpha.png        — topomapa 2D alpha HC vs Depresión

Uso:
  python eda_brmh.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne
import os

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH = 'EEG.machinelearing_data_BRMH.csv'   # ← ajusta la ruta si es necesario
OUT      = 'eda_outputs'
os.makedirs(OUT, exist_ok=True)

# Paleta de colores por trastorno
DISORDER_COLORS = {
    'Mood disorder':                        '#E05A3A',
    'Addictive disorder':                   '#F4A261',
    'Trauma and stress related disorder':   '#2E86DE',
    'Schizophrenia':                        '#8338EC',
    'Anxiety disorder':                     '#06D6A0',
    'Healthy control':                      '#4CAF50',
    'Obsessive compulsive disorder':        '#FFB703',
}

BANDS    = ['delta','theta','alpha','beta','highbeta','gamma']
BAND_HZ  = {'delta':2, 'theta':6, 'alpha':10, 'beta':20, 'highbeta':35, 'gamma':50}
BAND_LBL = ['Delta\n1-4 Hz','Theta\n4-8 Hz','Alpha\n8-13 Hz',
             'Beta\n13-30 Hz','High Beta\n30-40 Hz','Gamma\n40+ Hz']
BAND_COL = ['#cce5ff','#d4edda','#fff3cd','#f8d7da','#e2d9f3','#fde8d8']

# 19 canales estándar del dataset (en el orden de las letras a-s)
CHANS19  = ['FP1','FP2','F7','F3','Fz','F4','F8','T3','C3','Cz',
            'C4','T4','T5','P3','Pz','P4','T6','O1','O2']
CHANS14  = CHANS19[:14]   # primeros 14 para el heatmap
LETTER   = {c:l for l,c in zip('abcdefghijklmnopqrs', CHANS19)}

# ── Carga de datos ───────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['Unnamed: 122'], errors='ignore')
    print(f"Dataset cargado: {df.shape[0]} sujetos, {df.shape[1]} columnas")
    print(f"Trastornos: {df['main.disorder'].value_counts().to_dict()}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Distribución de clases (main.disorder)
# ════════════════════════════════════════════════════════════════════════════
def fig1_distribucion_clases(df):
    counts  = df['main.disorder'].value_counts()
    n_total = len(df)
    ideal   = n_total / len(counts)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[DISORDER_COLORS[d] for d in counts.index],
                  width=0.6, edgecolor='none', zorder=3)
    ax.axhline(ideal, color='black', lw=1.3, ls='--',
               label=f'Distribución ideal (1/{len(counts)})')
    for b, n in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                f'n={n}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Cantidad de Sujetos', fontsize=10)
    ax.set_title(f'1. Distribución de Clases — Dataset BRMH (n = {n_total})',
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xticklabels(counts.index, rotation=18, ha='right', fontsize=8.5)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, color='#eeeeee', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    path = f"{OUT}/fig1_distribucion_clases.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Potencia espectral por banda y trastorno (AB features reales)
# ════════════════════════════════════════════════════════════════════════════
def fig2_psd_por_banda(df):
    disorders = list(DISORDER_COLORS.keys())
    x = np.arange(len(disorders))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for ax, band, blbl, bcol in zip(axes, BANDS, BAND_LBL, BAND_COL):
        band_cols = [c for c in df.columns if f'.{band}.' in c and c.startswith('AB.')]
        means, errs = [], []
        for d in disorders:
            s = df[df['main.disorder'] == d][band_cols].mean(axis=1)
            means.append(s.mean()); errs.append(s.std())

        ax.bar(x, means, width=0.6,
               color=[DISORDER_COLORS[d] for d in disorders],
               edgecolor='none', zorder=3)
        ax.errorbar(x, means, yerr=errs, fmt='none',
                    color='#555', lw=1.2, capsize=3, zorder=4)
        ax.set_facecolor('#fafafa')
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace(' disorder', '').replace(' related disorder', '')
                            for d in disorders], rotation=35, ha='right', fontsize=7)
        ax.set_title(blbl, fontsize=9.5, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc=bcol, ec='none'))
        ax.set_ylabel('Potencia (µV²/Hz)', fontsize=8)
        ax.yaxis.grid(True, color='#eeeeee', lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    fig.suptitle('2. Potencia Espectral por Banda (AB features) — Dataset BRMH\n'
                 'Media ± DE por trastorno, 19 canales',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f"{OUT}/fig2_psd_por_banda.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Perfil espectral por trastorno (punto central de cada banda)
# ════════════════════════════════════════════════════════════════════════════
def fig3_perfil_espectral(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    for disorder, col in DISORDER_COLORS.items():
        sub    = df[df['main.disorder'] == disorder]
        freqs  = [BAND_HZ[b] for b in BANDS]
        vals   = []
        for band in BANDS:
            bc = [c for c in df.columns if f'.{band}.' in c and c.startswith('AB.')]
            vals.append(sub[bc].mean(axis=1).mean())
        ax.plot(freqs, vals, 'o-', color=col, lw=1.8, ms=6,
                label=disorder.replace(' disorder', '').replace(' related disorder', ''))

    for name, fl, fh, bc in [('δ',1,4,'#cce5ff'),('θ',4,8,'#d4edda'),
                               ('α',8,13,'#fff3cd'),('β',13,30,'#f8d7da'),
                               ('γ',30,55,'#e2d9f3')]:
        ax.axvspan(fl, fh, alpha=0.35, color=bc, zorder=0)
        ax.text((fl+fh)/2, 1, name, ha='center', fontsize=9,
                color='#555', transform=ax.get_xaxis_transform())

    ax.set_xlabel('Frecuencia representativa (Hz)', fontsize=10)
    ax.set_ylabel('Potencia media (µV²/Hz)', fontsize=10)
    ax.set_title('3. Perfil Espectral por Trastorno — Dataset BRMH\n'
                 'Potencia media por banda y diagnóstico',
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=2)
    ax.yaxis.grid(True, color='#eeeeee', lw=0.6)
    ax.xaxis.grid(True, color='#eeeeee', lw=0.6)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    path = f"{OUT}/fig3_perfil_espectral.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Heatmap coherencia Alpha 14×14 (COH features reales)
# ════════════════════════════════════════════════════════════════════════════
def fig4_heatmap_pearson(df):
    NC = len(CHANS14)

    def build_coh_matrix(df_sub, band='alpha'):
        band_letter = {'delta':'A','theta':'B','alpha':'C',
                       'beta':'D','highbeta':'E','gamma':'F'}[band]
        mat = np.full((NC, NC), np.nan)
        np.fill_diagonal(mat, 1.0)
        for i, ch1 in enumerate(CHANS14):
            for j, ch2 in enumerate(CHANS14):
                if i == j: continue
                a, b = (i, j) if i < j else (j, i)
                la, lb = LETTER[CHANS14[a]], LETTER[CHANS14[b]]
                col = f'COH.{band_letter}.{band}.{la}.{CHANS14[a]}.{lb}.{CHANS14[b]}'
                if col in df_sub.columns:
                    v = np.clip(df_sub[col].mean() / 100.0, -1, 1)
                    mat[i, j] = mat[j, i] = v
        return mat

    df_hc  = df[df['specific.disorder'] == 'Healthy control']
    df_dep = df[df['specific.disorder'] == 'Depressive disorder']
    mat_hc   = build_coh_matrix(df_hc)
    mat_dep  = build_coh_matrix(df_dep)
    mat_diff = mat_dep - mat_hc

    CMAP_CORR = LinearSegmentedColormap.from_list(
        'rdylgn', ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf',
                   '#d9ef8b','#a6d96a','#66bd63','#1a9850'])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    configs = [
        (mat_hc,   CMAP_CORR, 0,    1,   f'Healthy Control (n={len(df_hc)})',      '#4CAF50'),
        (mat_dep,  CMAP_CORR, 0,    1,   f'Depressive Disorder (n={len(df_dep)})', '#E05A3A'),
        (mat_diff, 'coolwarm',-0.4, 0.4, 'Diferencia  Depresión − HC',            '#333333'),
    ]
    for ax, (mat, cmap, vmin, vmax, title, tcol) in zip(axes, configs):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        for i in range(NC):
            for j in range(NC):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5,
                        color='white' if abs(v) > 0.6 else 'black')
        ax.set_xticks(range(NC)); ax.set_xticklabels(CHANS14, rotation=90, fontsize=8)
        ax.set_yticks(range(NC)); ax.set_yticklabels(CHANS14, fontsize=8)
        ax.set_title(title, color=tcol, fontsize=10, fontweight='bold', pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle('4. Heatmap Coherencia Alpha 14×14 (COH features reales)\n'
                 'Dataset BRMH — Banda Alpha (8–13 Hz)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f"{OUT}/fig4_heatmap_pearson.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Topomapa 2D banda Alpha — HC vs Depresión
# ════════════════════════════════════════════════════════════════════════════
def fig5_topomap_alpha(df):
    # Nombres de canales compatibles con MNE standard_1020
    CHANS_TOPO = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz',
                  'C4','T8','P7','P3','Pz','P4','P8','O1','O2']
    ALPHA_COLS = {cd: f'AB.C.alpha.{LETTER[cd]}.{cd}' for cd in CHANS19}

    info = mne.create_info(ch_names=CHANS_TOPO, sfreq=256, ch_types='eeg')
    info.set_montage('standard_1020', on_missing='ignore')

    NEURO = LinearSegmentedColormap.from_list(
        'neuro2', ['#053061','#2166ac','#4393c3','#92c5de',
                   '#f7f7f7','#fdbf6f','#e08214','#b35806','#7f3b08'])

    groups = {
        'Healthy Control':     (df[df['specific.disorder'] == 'Healthy control'],    '#4CAF50'),
        'Depressive Disorder': (df[df['specific.disorder'] == 'Depressive disorder'],'#E05A3A'),
    }
    all_vals = [np.array([sub[ALPHA_COLS[cd]].mean() for cd in CHANS19])
                for sub, _ in groups.values()]
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, (sub, col)), vals in zip(axes, groups.items(), all_vals):
        mne.viz.plot_topomap(vals, info, axes=ax, show=False,
                              cmap=NEURO, vlim=(vmin, vmax), contours=6, size=3)
        ax.set_title(f'{name}\n(n={len(sub)})', color=col,
                     fontsize=11, fontweight='bold', pad=8)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.016, 0.70])
    sm = plt.cm.ScalarMappable(cmap=NEURO, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Potencia Alpha (µV²/Hz)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle('5. Topomapa 2D — Banda Alpha (8–13 Hz)\n'
                 'Dataset BRMH — AB features reales, 19 canales',
                 fontsize=11, fontweight='bold')
    path = f"{OUT}/fig5_topomap_alpha.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f"  ✓ {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n═══ EDA EEG — Dataset BRMH ═══\n")
    df = load_data()
    fig1_distribucion_clases(df)
    fig2_psd_por_banda(df)
    fig3_perfil_espectral(df)
    fig4_heatmap_pearson(df)
    fig5_topomap_alpha(df)
    print(f"\n✅  5 figuras guardadas en ./{OUT}/\n")
