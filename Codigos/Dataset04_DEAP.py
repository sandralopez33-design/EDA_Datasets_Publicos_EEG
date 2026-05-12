"""
DEAP Dataset – EDA Visualizations (Academic/Scientific Style)
Subject s03 | 5 Figures for Biomedical Research
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import signal
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# IEEE / Academic matplotlib style
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Serif',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'axes.linewidth':     0.8,
    'grid.linewidth':     0.5,
    'lines.linewidth':    1.0,
    'axes.grid':          True,
    'grid.alpha':         0.35,
    'grid.color':         '#888888',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
})

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
with open('/mnt/user-data/uploads/s03.dat', 'rb') as f:
    raw = pickle.load(f, encoding='latin1')

EEG_RAW   = raw['data']      # (40, 40, 8064)  trials × channels × samples
LABELS     = raw['labels']   # (40, 4)          valence, arousal, dominance, liking
FS         = 128             # Hz
N_TRIALS   = 40
N_EEG      = 32              # first 32 channels are EEG
EEG        = EEG_RAW[:, :N_EEG, :]   # (40, 32, 8064)

# Skip the first 3-s baseline (384 samples): keep 60 s of stimulus
EEG_STIM   = EEG[:, :, 384:]         # (40, 32, 7680)

VALENCE    = LABELS[:, 0]
AROUSAL    = LABELS[:, 1]

# Channel names (DEAP standard 32-ch order)
CH_NAMES = [
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
    'Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz',
    'C4','T8','CP6','CP2','P4','P8','PO4','O2'
]

# Standard 10-20 2-D electrode positions (normalized, nose up)
CH_POS = {
    'Fp1':(-0.18, 0.85), 'Fp2':(0.18, 0.85),
    'AF3':(-0.31, 0.72), 'AF4':(0.31, 0.72),
    'F7': (-0.72, 0.46), 'F3': (-0.38, 0.52),
    'Fz': ( 0.00, 0.57), 'F4': ( 0.38, 0.52),
    'F8': ( 0.72, 0.46),
    'FC5':(-0.60, 0.24), 'FC1':(-0.22, 0.28),
    'FC2':( 0.22, 0.28), 'FC6':( 0.60, 0.24),
    'T7': (-0.83, 0.00), 'C3': (-0.42, 0.00),
    'Cz': ( 0.00, 0.00), 'C4': ( 0.42, 0.00),
    'T8': ( 0.83, 0.00),
    'CP5':(-0.60,-0.24), 'CP1':(-0.22,-0.28),
    'CP2':( 0.22,-0.28), 'CP6':( 0.60,-0.24),
    'P7': (-0.72,-0.46), 'P3': (-0.38,-0.52),
    'Pz': ( 0.00,-0.57), 'P4': ( 0.38,-0.52),
    'P8': ( 0.72,-0.46),
    'PO3':(-0.31,-0.70), 'PO4':( 0.31,-0.70),
    'O1': (-0.18,-0.83), 'Oz': ( 0.00,-0.87),
    'O2': ( 0.18,-0.83),
}

def band_power(sig, fs, fmin, fmax):
    """Mean band power via Welch PSD."""
    f, pxx = signal.welch(sig, fs, nperseg=fs*2)
    idx = (f >= fmin) & (f <= fmax)
    return np.mean(pxx[idx])

# ═══════════════════════════════════════════════════════════════
# Fig 1 – Bimodal histogram: Valence / Arousal distribution
# ═══════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(11, 3.8))
fig1.suptitle(
    'Fig. 1 – Distribución Emocional en el Espacio Valencia / Arousal  (S03, N=40 trials)',
    fontsize=11, fontweight='bold', y=1.01
)

bins = np.linspace(1, 9, 17)
colors_v = ['#2166ac', '#d73027']
colors_a = ['#1a9850', '#d73027']

# Valence histogram
ax = axes[0]
ax.hist(VALENCE, bins=bins, color='#2166ac', alpha=0.78, edgecolor='white',
        linewidth=0.6, label='Valence')
ax.axvline(5, color='#d73027', lw=1.2, ls='--', label='Threshold=5')
ax.axvline(VALENCE.mean(), color='#333333', lw=1.0, ls=':', label=f'Mean={VALENCE.mean():.2f}')
ax.set_xlabel('Valence Score (1–9 SAM Scale)')
ax.set_ylabel('Frequency (# Trials)')
ax.set_title('(a) Valence Distribution')
ax.legend(framealpha=0.5)

# Arousal histogram
ax = axes[1]
ax.hist(AROUSAL, bins=bins, color='#4dac26', alpha=0.78, edgecolor='white',
        linewidth=0.6, label='Arousal')
ax.axvline(5, color='#d73027', lw=1.2, ls='--', label='Threshold=5')
ax.axvline(AROUSAL.mean(), color='#333333', lw=1.0, ls=':', label=f'Mean={AROUSAL.mean():.2f}')
ax.set_xlabel('Arousal Score (1–9 SAM Scale)')
ax.set_title('(b) Arousal Distribution')
ax.legend(framealpha=0.5)

# 2-D scatter Valencia vs Arousal with quadrant
ax = axes[2]
low_v  = VALENCE < 5
low_a  = AROUSAL < 5
quadrant_colors = np.where(
    low_v & low_a,  0,
    np.where(~low_v & low_a,  1,
    np.where(low_v & ~low_a,  2, 3))
)
cmap_q = plt.cm.get_cmap('coolwarm', 4)
sc = ax.scatter(VALENCE, AROUSAL, c=quadrant_colors, cmap=cmap_q,
                s=55, alpha=0.85, edgecolors='grey', linewidths=0.4)
ax.axvline(5, color='grey', lw=0.8, ls='--')
ax.axhline(5, color='grey', lw=0.8, ls='--')
ax.set_xlabel('Valence')
ax.set_ylabel('Arousal')
ax.set_title('(c) 2D Affective Space (HVHA/HVLA/LVHA/LVLA)')
ax.text(6.5, 7.5, 'HVHA', fontsize=8, color='#b2182b', ha='center')
ax.text(6.5, 2.5, 'HVLA', fontsize=8, color='#4393c3', ha='center')
ax.text(2.5, 7.5, 'LVHA', fontsize=8, color='#d6604d', ha='center')
ax.text(2.5, 2.5, 'LVLA', fontsize=8, color='#4575b4', ha='center')

fig1.tight_layout()
fig1.savefig('/mnt/user-data/outputs/fig1_valence_arousal.png',
             dpi=300, bbox_inches='tight')
plt.close(fig1)
print("Fig 1 saved.")

# ═══════════════════════════════════════════════════════════════
# Fig 2 – Raw EEG occipital channel (O1, idx=13), trial 0
# ═══════════════════════════════════════════════════════════════
TRIAL     = 0
OCH_IDX   = 13    # O1
OCH_NAME  = 'O1'
eeg_raw   = EEG_STIM[TRIAL, OCH_IDX, :]   # 7680 samples = 60 s
t_vec     = np.arange(len(eeg_raw)) / FS

# Bandpass 1-45 Hz
b, a = signal.butter(4, [1, 45], btype='band', fs=FS)
eeg_filt = signal.filtfilt(b, a, eeg_raw)

fig2, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
fig2.suptitle(
    f'Fig. 2 – Señal EEG Cruda y Filtrada – Canal {OCH_NAME} (Trial {TRIAL+1})',
    fontsize=11, fontweight='bold'
)

axes[0].plot(t_vec, eeg_raw, color='#2c7bb6', lw=0.5, alpha=0.85)
axes[0].set_ylabel('Amplitude (μV)')
axes[0].set_title(f'(a) Raw EEG – {OCH_NAME}')
axes[0].yaxis.set_major_locator(ticker.MaxNLocator(5))

axes[1].plot(t_vec, eeg_filt, color='#1a9641', lw=0.5, alpha=0.85)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude (μV)')
axes[1].set_title(f'(b) Bandpass Filtered EEG (1–45 Hz) – {OCH_NAME}')
axes[1].yaxis.set_major_locator(ticker.MaxNLocator(5))

# Mark 10-s intervals
for ax in axes:
    for t_mark in range(10, 61, 10):
        ax.axvline(t_mark, color='#d7191c', lw=0.7, ls=':', alpha=0.6)

fig2.tight_layout()
fig2.savefig('/mnt/user-data/outputs/fig2_raw_eeg_O1.png',
             dpi=300, bbox_inches='tight')
plt.close(fig2)
print("Fig 2 saved.")

# ═══════════════════════════════════════════════════════════════
# Fig 3 – Alpha-band topomap (manual interpolation)
# ═══════════════════════════════════════════════════════════════
# Compute mean alpha power per channel across all trials
alpha_power = np.zeros(N_EEG)
for ch in range(N_EEG):
    pows = [band_power(EEG_STIM[tr, ch, :], FS, 8, 13) for tr in range(N_TRIALS)]
    alpha_power[ch] = np.mean(pows)

# Positions
xs = np.array([CH_POS[ch][0] for ch in CH_NAMES])
ys = np.array([CH_POS[ch][1] for ch in CH_NAMES])

# Grid for interpolation
grid_res = 300
xi = np.linspace(-1.0, 1.0, grid_res)
yi = np.linspace(-1.0, 1.0, grid_res)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata((xs, ys), alpha_power, (Xi, Yi), method='cubic')

# Mask outside head circle
R = 0.92
mask = (Xi**2 + Yi**2) > R**2
Zi[mask] = np.nan

fig3, ax = plt.subplots(figsize=(5.5, 6))
fig3.suptitle(
    'Fig. 3 – Topomapa EEG: Potencia Banda Alpha (8–13 Hz)\n'
    '(Media 40 trials, Sistema 10-20, S03)',
    fontsize=11, fontweight='bold'
)

im = ax.contourf(Xi, Yi, Zi, levels=256, cmap='viridis', extend='both')
ax.contour(Xi, Yi, Zi, levels=10, colors='white', linewidths=0.3, alpha=0.4)

# Head circle
head_circle = plt.Circle((0, 0), R, fill=False, color='black', lw=1.5)
ax.add_patch(head_circle)

# Nose
nose_x = [-.07, 0, .07]
nose_y = [R, R + 0.10, R]
ax.plot(nose_x, nose_y, 'k-', lw=1.5)

# Ears
for sign in [-1, 1]:
    ear = Ellipse((sign * (R + 0.04), 0), 0.08, 0.15,
                  fill=False, color='black', lw=1.2)
    ax.add_patch(ear)

# Electrodes
ax.scatter(xs, ys, c='white', s=28, edgecolors='black', linewidths=0.7, zorder=5)
for i, name in enumerate(CH_NAMES):
    ax.text(xs[i], ys[i] + 0.07, name, ha='center', va='bottom',
            fontsize=6.5, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.05', fc='none', ec='none'))

cb = fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Power Spectral Density (μV²/Hz)', fontsize=9)
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.20)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(a) Alpha Power Distribution – All Channels', fontsize=9, pad=4)

fig3.tight_layout()
fig3.savefig('/mnt/user-data/outputs/fig3_topomap_alpha.png',
             dpi=300, bbox_inches='tight')
plt.close(fig3)
print("Fig 3 saved.")

# ═══════════════════════════════════════════════════════════════
# Fig 4 – Pearson inter-channel correlation heatmap
# ═══════════════════════════════════════════════════════════════
# Use all trials concatenated along time axis
eeg_concat = EEG_STIM.transpose(1, 0, 2).reshape(N_EEG, -1)  # (32, 40*7680)

# Bandpass 1-45 Hz for cleaner correlation
eeg_bp = np.zeros_like(eeg_concat)
for ch in range(N_EEG):
    eeg_bp[ch] = signal.filtfilt(b, a, eeg_concat[ch])

corr_mat = np.corrcoef(eeg_bp)  # (32, 32)

fig4, ax = plt.subplots(figsize=(9, 8))
fig4.suptitle(
    'Fig. 4 – Matriz de Correlación de Pearson Inter-Canal EEG\n'
    '(Señal Filtrada 1–45 Hz, 40 Trials Concatenados, S03)',
    fontsize=11, fontweight='bold'
)

mask_tri = np.zeros_like(corr_mat, dtype=bool)
# Show full matrix
sns.heatmap(
    corr_mat, ax=ax,
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    xticklabels=CH_NAMES, yticklabels=CH_NAMES,
    linewidths=0.15, linecolor='#cccccc',
    annot=False,
    cbar_kws={'label': 'Pearson Correlation Coefficient (r)', 'shrink': 0.82}
)
ax.set_xticklabels(CH_NAMES, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(CH_NAMES, rotation=0, fontsize=8)
ax.set_title('(a) Functional Connectivity – Pearson r Matrix', fontsize=9, pad=6)

# Annotate brain regions
region_bounds = [
    (0,  7,  'Frontal'),
    (8,  11, 'Central'),
    (12, 15, 'Parietal-Occ. L'),
    (16, 22, 'Frontal R'),
    (23, 27, 'Central R'),
    (28, 31, 'Occipital'),
]
for s, e, lbl in region_bounds:
    ax.add_patch(plt.Rectangle((s, s), e-s+1, e-s+1,
                               fill=False, edgecolor='black', lw=1.4, zorder=10))

fig4.tight_layout()
fig4.savefig('/mnt/user-data/outputs/fig4_correlation_heatmap.png',
             dpi=300, bbox_inches='tight')
plt.close(fig4)
print("Fig 4 saved.")

# ═══════════════════════════════════════════════════════════════
# Fig 5 – Spectrogram: full 60-s epoch, O1 channel
# ═══════════════════════════════════════════════════════════════
eeg_spec = EEG_STIM[0, OCH_IDX, :]   # O1, trial 0
eeg_spec_filt = signal.filtfilt(b, a, eeg_spec)

fig5, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 2.5]})
fig5.suptitle(
    f'Fig. 5 – Espectrograma Tiempo-Frecuencia – Canal {OCH_NAME} (Trial 1, S03)\n'
    '(Short-Time Fourier Transform | Ventana Hann, NFFT=256)',
    fontsize=11, fontweight='bold'
)

# Top: filtered time series
t_v = np.arange(len(eeg_spec_filt)) / FS
axes[0].plot(t_v, eeg_spec_filt, color='#1f78b4', lw=0.6, alpha=0.9)
axes[0].set_ylabel('Amplitude (μV)')
axes[0].set_title(f'(a) Filtered EEG – {OCH_NAME} (1–45 Hz)', fontsize=9)
axes[0].yaxis.set_major_locator(ticker.MaxNLocator(4))
axes[0].set_xlim(0, 60)

# Bottom: spectrogram
f_spec, t_spec, Sxx = signal.spectrogram(
    eeg_spec_filt, fs=FS, window='hann',
    nperseg=256, noverlap=224, nfft=512,
    scaling='density'
)
# Limit to 0-50 Hz
freq_mask = f_spec <= 50
f_plot = f_spec[freq_mask]
Sxx_plot = Sxx[freq_mask, :]
Sxx_db = 10 * np.log10(Sxx_plot + 1e-12)

im5 = axes[1].pcolormesh(t_spec, f_plot, Sxx_db,
                          cmap='plasma', shading='gouraud',
                          vmin=np.percentile(Sxx_db, 5),
                          vmax=np.percentile(Sxx_db, 98))
cb5 = fig5.colorbar(im5, ax=axes[1], pad=0.02)
cb5.set_label('Power (dB re μV²/Hz)', fontsize=9)

# EEG band markers
bands = [('Delta',0.5,4,'#ffffb2'),('Theta',4,8,'#fecc5c'),
         ('Alpha',8,13,'#fd8d3c'),('Beta',13,30,'#f03b20'),('Gamma',30,50,'#bd0026')]
for name, lo, hi, col in bands:
    axes[1].axhspan(lo, hi, alpha=0.08, color=col)
    axes[1].text(61, (lo+hi)/2, name, va='center', fontsize=7.5,
                 color='#333333', fontweight='bold')

axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('(b) STFT Spectrogram – Temporal-Spectral EEG Dynamics', fontsize=9)
axes[1].set_xlim(0, 60)
axes[1].set_ylim(0, 50)

# Synchronize x-axis
axes[0].set_xlim(0, 60)

fig5.tight_layout(h_pad=1.5)
fig5.savefig('/mnt/user-data/outputs/fig5_spectrogram_O1.png',
             dpi=300, bbox_inches='tight')
plt.close(fig5)
print("Fig 5 saved.")

print("\n✓ All 5 figures saved to /mnt/user-data/outputs/")
