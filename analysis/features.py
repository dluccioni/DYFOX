"""Image moments for classifying dislocation type and Burgers-vector sign.

    features       five moments about a centre it locates itself,
                   radii from the crop centre, pitch in metres
    rich_features  eleven moments on a pre-centred crop, radii from the
                   intensity centroid, pitch in micrometres
    locate         the centre finder both use: centroid of the smoothed
                   top-30 % region
    loo_accuracy   leave-one-out classification accuracy
"""

import numpy as np
from scipy.ndimage import gaussian_filter

FEATURE_NAMES = ["log10 core/halo", "anisotropy", "rms radius (um)",
                 "|A2|/A0", "|A4|/A0"]

RICH_FEATURE_NAMES = [
    "log10 core/halo", "anisotropy", "rms radius (um)",
    "|A2|/A0", "|A4|/A0", "|A1|/A0", "|A2 inner|/A0 inner",
    "log10 ann0/ann1", "log10 ann1/ann2", "log10 ann2/ann3",
    "log10 peak/median"]


def crop_about(img, cy_px, cx_px, h_px):
    return img[cy_px - h_px:cy_px + h_px, cx_px - h_px:cx_px + h_px]


def locate(img):
    """Pattern centre: centroid of the smoothed top-30% region."""
    sm = gaussian_filter(np.asarray(img, dtype=float), 8)
    m = sm > 0.3 * sm.max()
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    w = sm * m
    w = w / w.sum()
    return int(round((w * yy).sum())), int(round((w * xx).sum()))


def features(img, dx, crop_half_um=12.0, center=None):
    """Five rotation-aware moments about the located centre.

    In order: log10 core/halo, axis anisotropy, rms radius, and |A2|/A0
    and |A4|/A0 of the halo's azimuthal intensity profile.

    `dx` is the pixel pitch in metres, not micrometres. The micrometre
    grid is then built as `* dx * 1e6`, in that order, which matters in
    the last bit.
    """
    cy, cx = locate(img) if center is None else center
    h = int(crop_half_um / (dx * 1e6))
    cy = min(max(cy, h), img.shape[0] - h)
    cx = min(max(cx, h), img.shape[1] - h)
    c = crop_about(img, cy, cx, h).astype(float)
    yy, xx = np.mgrid[0:c.shape[0], 0:c.shape[1]]
    x_um = (xx - c.shape[1] / 2) * dx * 1e6
    y_um = (yy - c.shape[0] / 2) * dx * 1e6
    r = np.hypot(x_um, y_um)
    th = np.arctan2(x_um, y_um)
    core = c[r < 1.5].mean()
    halo_m = (r > 2.5) & (r < 8)
    halo = c[halo_m].mean()
    m = r < 10
    wm = np.clip(c * m, 0, None)
    tot = wm.sum()
    if tot <= 0:
        return np.zeros(5)
    wm = wm / tot
    cyw = (wm * y_um).sum()
    cxw = (wm * x_um).sum()
    var_y = (wm * (y_um - cyw) ** 2).sum()
    var_x = (wm * (x_um - cxw) ** 2).sum()
    w_h = np.clip(c[halo_m], 0, None)
    A0 = w_h.sum()
    A2 = np.abs((w_h * np.exp(-2j * th[halo_m])).sum())
    A4 = np.abs((w_h * np.exp(-4j * th[halo_m])).sum())
    return np.array([np.log10(max(core, 1e-12) / max(halo, 1e-12)),
                     (var_x - var_y) / (var_x + var_y),
                     np.sqrt(var_x + var_y),
                     A2 / max(A0, 1e-12), A4 / max(A0, 1e-12)])


def rich_features(img, dx_um):
    """Eleven image moments about the intensity centroid.

    `dx_um` is the pixel pitch in MICROMETRES (the crop is already
    centred, so no crop step and no metre-to-micrometre product).
    """
    img = np.asarray(img, float)
    H = img.shape[0]
    yy, xx = np.mgrid[0:H, 0:img.shape[1]]
    sm = gaussian_filter(img, 8)
    m0 = sm > 0.3 * sm.max()
    w = sm * m0
    w = w / w.sum()
    cy = (w * yy).sum()
    cx = (w * xx).sum()
    x_um = (xx - cx) * dx_um
    y_um = (yy - cy) * dx_um
    r = np.hypot(x_um, y_um)
    th = np.arctan2(x_um, y_um)
    c = np.clip(img, 0, None)
    ann = [c[(r >= r0) & (r < r1)].mean() + 1e-12
           for r0, r1 in [(0, 1.5), (1.5, 3), (3, 6), (6, 10)]]
    halo_m = (r > 2.5) & (r < 8)
    w_h = c[halo_m]
    A0 = w_h.sum() + 1e-12
    A1 = np.abs((w_h * np.exp(-1j * th[halo_m])).sum())
    A2 = np.abs((w_h * np.exp(-2j * th[halo_m])).sum())
    A4 = np.abs((w_h * np.exp(-4j * th[halo_m])).sum())
    in_m = (r > 1) & (r < 4)
    w_i = c[in_m]
    A0i = w_i.sum() + 1e-12
    A2i = np.abs((w_i * np.exp(-2j * th[in_m])).sum())
    sig = r < 10
    wm = c * sig
    wm = wm / wm.sum()
    cyw = (wm * y_um).sum()
    cxw = (wm * x_um).sum()
    var_y = (wm * (y_um - cyw) ** 2).sum()
    var_x = (wm * (x_um - cxw) ** 2).sum()
    return np.array([
        np.log10(ann[0] / ann[2]),
        (var_x - var_y) / (var_x + var_y),
        np.sqrt(var_x + var_y),
        A2 / A0, A4 / A0, A1 / A0, A2i / A0i,
        np.log10(ann[0] / ann[1]), np.log10(ann[1] / ann[2]),
        np.log10(ann[2] / ann[3]),
        np.log10(c.max() / (np.median(c) + 1e-12)),
    ])


def loo_accuracy(F, labels_fn):
    """Leave-one-out nearest-centroid accuracy on standardized features.

    F has shape (n_classes, n_realizations, n_features).  `labels_fn`
    maps a class index to the label being predicted: the identity for the
    full classification, `lambda c: c % 2` for sign only.
    """
    n_classes, n_ens, nf = F.shape
    mu = F.reshape(-1, nf).mean(axis=0)
    sd = F.reshape(-1, nf).std(axis=0) + 1e-12
    Fz = (F - mu) / sd
    correct = 0
    for ci in range(n_classes):
        for j in range(n_ens):
            cents = []
            for ck in range(n_classes):
                sel = Fz[ck] if ck != ci else np.delete(Fz[ck], j, axis=0)
                cents.append(sel.mean(axis=0))
            pred = int(np.argmin([np.sum((Fz[ci, j] - cc) ** 2)
                                  for cc in cents]))
            correct += int(labels_fn(pred) == labels_fn(ci))
    return correct / (n_classes * n_ens)


def standardise(F):
    """Mean and standard deviation used to whiten a feature bank."""
    nf = F.shape[-1]
    flat = F.reshape(-1, nf)
    return flat.mean(axis=0), flat.std(axis=0) + 1e-12


def nearest_centroid(f, centroids):
    """Index of the nearest centroid to a standardized feature vector."""
    return int(np.argmin([np.sum((f - cc) ** 2) for cc in centroids]))
