"""The grid a Laue solve runs on, and the depth step it implies.

`LaueGrid` holds the crystal thickness, the pixel pitch and the beam
shape.

    grid = LaueGrid(xtal, t_crystal=273e-6, dx=0.25e-6, beam_width=120e-6)

Unless `Nz` is given, the depth step is

    dz = dx / tan(theta_B)

which advances a characteristic of the Borrmann fan exactly one pixel
per step, so transport is an index shift rather than an interpolation.
`grid.exact` records whether that holds; passing `Nz` sets the step
independently and may make it false.
"""

import numpy as np


class LaueGrid:
    """Grid and beam settings for one solve.

    `beam_thickness` is the top-hat sheet thickness in metres. Setting
    `beam_sigma` instead gives a Gaussian amplitude profile of that rms
    width, whose intensity FWHM is 2*sqrt(ln 2) times it. `grid_pad` is
    the margin in pixels beyond the Borrmann fan.
    """

    def __init__(self, xtal, t_crystal, dx, beam_width, *,
                 beam_thickness=1.0e-6, beam_sigma=0.0, Nz=None,
                 grid_pad=30):
        self.xtal = xtal
        self.beam_thickness = float(beam_thickness)
        self.beam_sigma = float(beam_sigma)
        self.dx = float(dx)
        self.beam_width = float(beam_width)
        self.grid_pad = int(grid_pad)
        if Nz is None:
            dz = self.dx / xtal.tan_tB
            self.Nz = int(round(float(t_crystal) / dz)) + 1
            self.t_crystal = (self.Nz - 1) * dz
            self.exact = True
        else:
            self.Nz = int(Nz)
            self.t_crystal = float(t_crystal)
            dz = self.t_crystal / (self.Nz - 1)
            self.exact = abs(xtal.tan_tB * dz / self.dx - 1.0) < 1e-9

    def replace(self, **kw):
        """A copy with some settings changed."""
        base = dict(xtal=self.xtal, t_crystal=self.t_crystal, dx=self.dx,
                    beam_width=self.beam_width,
                    beam_thickness=self.beam_thickness,
                    beam_sigma=self.beam_sigma, grid_pad=self.grid_pad)
        if "Nz" not in kw and not self.exact:
            base["Nz"] = self.Nz
        base.update(kw)
        return LaueGrid(**base)

    def sheet_arg(self):
        """The kernel's beam parameter: half-width, or minus the sigma."""
        if self.beam_sigma > 0:
            return -self.beam_sigma
        return self.beam_thickness / 2.0

    def geometry(self):
        """(Nx, Ny, dz, ds, shift_px) for this configuration.

        `ds` is the step along the characteristic rather than along z,
        and `shift_px` the per-step pixel shift, snapped to an integer
        when within round-off of one.
        """
        params = self.xtal
        borrmann = self.t_crystal * params.tan_tB
        pad = self.grid_pad * self.dx
        eff_thick = (6.0 * self.beam_sigma if self.beam_sigma > 0
                     else self.beam_thickness)
        extent_x = eff_thick + 2 * borrmann + 2 * pad
        Nx = max(int(np.ceil(extent_x / self.dx)), 128)
        Nx = int(np.ceil(Nx / 32)) * 32
        Ny = max(int(np.ceil((self.beam_width + 2 * pad) / self.dx)), 64)
        Ny = int(np.ceil(Ny / 32)) * 32
        dz = self.t_crystal / (self.Nz - 1)
        ds = dz / params.cos_tB
        shift_px = params.tan_tB * dz / self.dx
        if abs(shift_px - round(shift_px)) < 1e-9:
            shift_px = float(round(shift_px))
        return Nx, Ny, dz, ds, shift_px

    @property
    def shape(self):
        """(Ny, Nx), the shape of one exit wave."""
        Nx, Ny = self.geometry()[:2]
        return Ny, Nx

    def crop(self, half_um=25.0):
        """Slices and display extent for a crop about the grid centre."""
        from analysis.metrics import crop_slices
        Nx, Ny = self.geometry()[:2]
        return crop_slices(Nx, Ny, self.dx, half_um)

    def key(self):
        """The identity of this grid, for a cache key."""
        return dict(beam_thickness=self.beam_thickness,
                    beam_sigma=self.beam_sigma,
                    t_crystal=self.t_crystal, Nz=self.Nz, dx=self.dx,
                    beam_width=self.beam_width, grid_pad=self.grid_pad)

    def __repr__(self):
        Nx, Ny = self.geometry()[:2]
        return (f"LaueGrid({Ny}x{Nx}, Nz={self.Nz}, "
                f"t={self.t_crystal * 1e6:.1f}um, "
                f"dx={self.dx * 1e9:.0f}nm"
                + ("" if self.exact else ", interpolating") + ")")
