import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, writers
from matplotlib.colors import ListedColormap
import seaborn as sns
from skill_metrics import taylor_diagram, taylor_statistics

EUROPE = [-30, 40, 34.25, 72]


def animate_choropleth(lon, lat, data, times, species_idx, out_gif):
    """
    data : [T, H, W] for a single species.
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': proj})
    ax.set_extent([-30, 40, 34.25, 72], crs=proj)
    ax.coastlines(resolution="50m")

    mesh = ax.pcolormesh(lon, lat, data[0], cmap='viridis', transform=proj)
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical')
    cbar.set_label('Species count')
    ttl = ax.set_title("")

    def update(frame):
        mesh.set_array(data[frame].ravel())
        ttl.set_text(f"Species {species_idx} – {times[frame]}")
        return mesh, ttl

    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(out_gif, writer=writers['ffmpeg'](fps=3))

def hovmoller(lat, times, data, ax=None, title="Hovmöller Diagram"):
    """
    data : [T, H] lat-mean or lon-mean collapsed map.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.contourf(times, lat, data.T, levels=60, cmap='viridis')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Time')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Species count')

def make_taylor(ref, models, labels, ax=None):
    """
    ref    : [N] reference series flattened (truth)
    models : list of [N] model series flattened
    """
    stats = []
    for m in models:
        std  = np.std(m, ddof=1)
        corr = np.corrcoef(ref, m)[0, 1]
        stats.append((std, corr))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    td, _, _, _ = taylor_diagram(np.std(ref, ddof=1), fig=ax.figure, rect=111,
                                      label='REF')
    for (std, corr), lab in zip(stats, labels):
        td.add_sample(std, corr, marker='o', label=lab)
    td.add_legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

def plot_change_map(lat, lon, y0, y1_obs, y1_pred, species_id, extent=EUROPE):
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(1, 4, figsize=(18, 4),
                            subplot_kw={'projection': proj})
    titles = [f'Year-0 obs', 'Year-1 obs', 'Year-1 pred',
              'Pred - obs (yr-1)']
    mats   = [y0, y1_obs, y1_pred, y1_pred - y1_obs]
    for ax, mat, ttl in zip(axs, mats, titles):
        ax.set_extent(extent, crs=proj);  ax.coastlines('50m')
        mesh = ax.pcolormesh(lon, lat, mat, cmap='viridis',
                             transform=proj)
        ax.set_title(ttl);  plt.colorbar(mesh, ax=ax, shrink=.6)
    fig.suptitle(f'Species {species_id}')
    plt.tight_layout()  
    plt.show()

def plot_hexbin(all_pred, all_obs):
    fig, ax = plt.subplots(figsize=(5, 5))
    hb = ax.hexbin(all_obs, all_pred, gridsize=80, bins='log',
                   mincnt=1, cmap='viridis')
    ax.plot([0, all_obs.max()], [0, all_obs.max()], 'r--', lw=1)
    ax.set_xlabel('Observed');  ax.set_ylabel('Predicted')
    ax.set_title('Calibration density')
    cb = fig.colorbar(hb, ax=ax); cb.set_label('log10(N)')
    plt.show()

def plot_taylor(lat, lon, y1_obs, y1_pred):
    ref_std = np.std(y1_obs)
    fig = plt.figure(figsize=(6, 6))
    td  = taylor_diagram.TaylorDiagram(ref_std, fig=fig, label='REF')
    std  = np.std(y1_pred);  corr = np.corrcoef(y1_obs.ravel(),
                                                y1_pred.ravel())[0, 1]
    td.add_sample(std, corr, marker='o', label='Model')
    td.add_legend(loc='upper right', fontsize='small')
    plt.title('Spatial pattern skill'); plt.show()

def plot_confusion_map(lat, lon, y_true, y_pred, thresh=1):
    cats = (y_pred >= thresh).astype(int) + 2 * (y_true >= thresh).astype(int)
    cmap = ListedColormap(['white', 'royalblue', 'orangered', 'limegreen'])
    labels = ['TN', 'FP', 'FN', 'TP']
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={'projection': proj})
    ax.set_extent(EUROPE, crs=proj);  ax.coastlines('50m')
    mesh = ax.pcolormesh(lon, lat, cats, cmap=cmap, transform=proj,
                         vmin=0, vmax=3)
    cb = fig.colorbar(mesh, ticks=[0.5,1.5,2.5,3.5], shrink=.7)
    cb.ax.set_yticklabels(labels); ax.set_title('Presence/absence confusion')
    plt.show()

def plot_error_violin(errors_per_species):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.violinplot(data=errors_per_species.T, cut=0, bw=.2, ax=ax,
                   inner='quartile'); ax.set_xlabel('Species index')
    ax.set_ylabel('|Pred - Obs|'); ax.set_title('Error distribution per species')
    plt.show()

def plot_taylor_single(obs: np.ndarray, pred: np.ndarray,
                       title: str = "Taylor diagram"):
    """
    obs, pred : 1-D arrays of the same length (flatten y-lat-lon grid).
    """
    stats = taylor_statistics(pred, obs) # returns dict of scalars

    STDs = np.array([stats['sdev']])# model std‑dev(s)
    RMSs = np.array([stats['crmsd']]) # centred‑RMS diff(s)
    CORs = np.array([stats['ccoef']]) # correlation(s)

    fig = plt.figure(figsize=(6, 6))
    taylor_diagram(STDs, RMSs, CORs,
                   markerLabel= ['Model-1'],
                   markerColor = 'royalblue',
                   markerSize = 8,
                   colCOR = 'black',
                   titleOBS = title,
                   styleOBS = '-k',           # ref point style
                   markerLabelColor = 'royalblue',
                #    fig = fig
                   )
    plt.show()