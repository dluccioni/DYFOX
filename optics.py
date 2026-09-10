"""The imaging stage: an objective, a pupil, and a spiral phase plate.

An `Optics` holds a numerical aperture, a focal length and a crystal,
and hands out pupils.

    lens = Optics(NA=1e-4, f_lens=0.274, xtal=xtal)
    open_img = lens.image(stack, dx, ell=0)
    spiral = lens.image(stack, dx, ell=+2)

A pupil of charge `ell` carries a spiral phase exp(i ell phi) inside the
NA disc. Metres in the back focal plane map to spatial frequencies
through x_bfp = lambda f fx, which is where `f_lens` enters.

The module-level `object_spectrum`, `image_from_spectrum`,
`apply_pupils` and `image_intensity` are the imaging primitives the
methods are built from.
"""

import numpy as np

from backend import FP64, array_module, cupy


def _modules(xp, precision):
    """(xp, real dtype, complex dtype) for whichever library is in use."""
    if xp is None:
        xp = cupy()
    if xp is np:
        return xp, precision.real, precision.complex
    return xp, precision.cp_real, precision.cp_complex


class Optics:
    """An objective of given numerical aperture, viewing a given crystal.

    `xp` picks the array library, CuPy by default. Pupils are cached on
    every argument that shapes them.
    """

    def __init__(self, NA, f_lens, xtal, *, precision=FP64, xp=None):
        self.NA = float(NA)
        self.f_lens = float(f_lens)
        self.xtal = xtal
        self.precision = precision
        self.xp = xp
        self._cache = {}

    @property
    def aperture_f(self):
        """Radius of the NA disc in spatial frequency."""
        return self.NA / self.xtal.lam

    def resolution(self):
        """Rayleigh resolution, as a rough sanity number in metres."""
        return 0.61 * self.xtal.lam / self.NA

    def pupil(self, Nx, Ny, dx, ell=0, *, aperture_f=(0.0, 0.0),
              singularity_f=(0.0, 0.0), dc_zero="exact", levels=0,
              cache=True):
        """The pupil on the fftshifted frequency grid.

        `aperture_f` and `singularity_f` are frequency offsets of the
        aperture stop and the spiral singularity. `levels` quantises the
        spiral phase into that many steps.

        `dc_zero` decides the bin at the origin:

            "exact"    zero it when the singularity is exactly centred
            "half_bin" zero it when within half a bin
            "never"    leave it alone

        `cache=False` skips the pupil cache.
        """
        xp, real_t, cplx_t = _modules(self.xp, self.precision)
        ax, ay = float(aperture_f[0]), float(aperture_f[1])
        sx, sy = float(singularity_f[0]), float(singularity_f[1])
        key = (Nx, Ny, float(dx), int(ell), ax, ay, sx, sy,
               dc_zero, int(levels), xp is np, self.precision.name,
               self.NA, self.xtal.lam)
        if cache and key in self._cache:
            return self._cache[key]

        fx = xp.fft.fftshift(xp.fft.fftfreq(Nx, dx))
        fy = xp.fft.fftshift(xp.fft.fftfreq(Ny, dx))
        FX, FY = xp.meshgrid(fx, fy)
        if ax == 0.0 and ay == 0.0:
            rad = xp.sqrt(FX ** 2 + FY ** 2)
        elif ay == 0.0:
            rad = xp.sqrt((FX - ax) ** 2 + FY ** 2)
        else:
            rad = xp.sqrt((FX - ax) ** 2 + (FY - ay) ** 2)
        circ = (rad <= self.aperture_f).astype(real_t)
        if ell == 0:
            pupil = circ.astype(cplx_t)
        else:
            phi = xp.arctan2(FY - sy, FX - sx)
            ph = ell * phi
            if levels:
                step = 2 * np.pi / levels
                ph = xp.floor(xp.mod(ph, 2 * np.pi) / step) * step
            pupil = (circ * xp.exp(1j * ph)).astype(cplx_t)
            if self._zero_dc(dc_zero, sx, sy, fx):
                pupil[Ny // 2, Nx // 2] = 0.0
        if cache:
            self._cache[key] = pupil
        return pupil

    @staticmethod
    def _zero_dc(policy, sx, sy, fx):
        if policy == "never":
            return False
        if policy == "exact":
            return sx == 0.0 and sy == 0.0
        if policy == "half_bin":
            step = float(fx[1] - fx[0])
            return abs(sx) < 0.5 * step and abs(sy) < 0.5 * step
        raise ValueError(f"dc_zero is 'exact', 'half_bin' or 'never', "
                         f"not {policy!r}")

    def spiral(self, Nx, Ny, dx, ell=0, offset_bfp=(0.0, 0.0), levels=0,
               cache=True):
        """Pupil with the plate displaced by `offset_bfp` metres.

        Moves the singularity; the aperture stays put.
        """
        dfx = offset_bfp[0] / (self.xtal.lam * self.f_lens)
        dfy = offset_bfp[1] / (self.xtal.lam * self.f_lens)
        return self.pupil(Nx, Ny, dx, ell, singularity_f=(dfx, dfy),
                          dc_zero="exact", levels=levels, cache=cache)

    def chromatic(self, Nx, Ny, dx, ell, delta_E, levels=0,
                  dc_zero="never", cache=True):
        """Pupil seen by photons at a relative energy offset delta_E.

        Aperture and singularity both slide by
        sin(2 theta_B) delta_E / lambda. Energies whose walk exceeds the
        NA disc vignette away entirely.
        """
        dfx = self.xtal.sin_2tB * delta_E / self.xtal.lam
        return self.pupil(Nx, Ny, dx, ell, aperture_f=(dfx, 0.0),
                          singularity_f=(dfx, 0.0), dc_zero=dc_zero,
                          levels=levels, cache=cache)

    def image(self, stack, dx, ell=0, offset_bfp=(0.0, 0.0), levels=0):
        """Image intensity of a stack of exit waves through one pupil."""
        Ny, Nx = stack.shape[-2:]
        pup = self.spiral(Nx, Ny, dx, ell, offset_bfp, levels)
        return image_intensity(stack, pup)

    def images(self, field, dx, ells=(0, 2), offset_bfp=(0.0, 0.0)):
        """One exit wave through several charges, sharing one spectrum."""
        Ny, Nx = field.shape[-2:]
        pups = [self.spiral(Nx, Ny, dx, e, offset_bfp) for e in ells]
        return apply_pupils(field, pups)

    def clear_cache(self):
        """Forget the cached pupils. Only sweeps over NA need this."""
        self._cache.clear()

    def __repr__(self):
        return (f"Optics(NA={self.NA:g}, f={self.f_lens:g} m, "
                f"{self.xtal.label}, {int(self.resolution() * 1e9)} nm)")


def object_spectrum(field):
    """Spectrum of an exit wave, on the fftshifted grid.

    Worth computing once when several pupils are to be applied to it,
    which is what `apply_pupils` does.
    """
    xp = array_module(field)
    if field.ndim == 2:
        return xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(field)))
    sh = xp.fft.fftshift(field, axes=(-2, -1))
    return xp.fft.fftshift(xp.fft.fft2(sh, axes=(-2, -1)), axes=(-2, -1))


def image_from_spectrum(spec, pupil):
    """Intensity in the image plane, given a spectrum and a pupil."""
    xp = array_module(spec)
    if spec.ndim == 2:
        img = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(spec * pupil)))
        return xp.abs(img) ** 2
    img = xp.fft.ifftshift(spec * pupil, axes=(-2, -1))
    img = xp.fft.ifft2(img, axes=(-2, -1))
    img = xp.fft.fftshift(img, axes=(-2, -1))
    return xp.abs(img) ** 2


def apply_pupils(field, pupils):
    """One exit wave through many pupils, sharing a single forward transform.

    Returns (n_pupils, Ny, Nx).
    """
    xp = array_module(field)
    spec = object_spectrum(field)
    stack = xp.stack([spec * pup for pup in pupils])
    img = xp.fft.ifftshift(stack, axes=(-2, -1))
    img = xp.fft.ifft2(img, axes=(-2, -1))
    img = xp.fft.fftshift(img, axes=(-2, -1))
    return xp.abs(img) ** 2


def image_intensity(stack, pupil):
    """A stack of exit waves through one pupil, batched."""
    return image_from_spectrum(object_spectrum(stack), pupil[None]
                               if stack.ndim == 3 else pupil)
