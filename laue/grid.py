"""The grid a Laue solve runs on, and the depth step it implies.

`LaueGrid` is a value object: crystal thickness, pixel pitch, and how
the beam is shaped. Build one and pass it to the solver.

    grid = LaueGrid(xtal, t_crystal=273e-6, dx=0.25e-6, beam_width=120e-6)

The one thing worth understanding before changing anything here is the
depth step. It is not a free parameter. It is set to

    dz = dx / tan(theta_B)

so that a characteristic of the Borrmann fan advances exactly one pixel
per depth step, and transporting the field is an index shift with no
interpolation at all.

Interpolate instead, at some fraction of a pixel per step, and the
error compounds over a thousand steps into a low-pass filter a few
micrometres wide. On a micrometre-scale weak-beam dislocation signal
that costs a factor of about 65 in the peak. The earlier pipeline ran
at 0.36 pixels per step and lost exactly that.

So refine resolution through `dx`, which carries `dz` with it. Passing
`Nz` yourself breaks the tie deliberately, which is worth doing only if
what you are studying is the interpolating transport itself.
"""

import numpy as np


class LaueGrid:
    """Grid and beam settings for one solve.

    `beam_thickness` is the top-hat sheet thickness in metres. Setting
    `beam_sigma` instead replaces the top hat with a Gaussian amplitude
    profile of that rms width, whose intensity full width at half
    maximum is 2*sqrt(ln 2) times it.

    `grid_pad` is how many pixels of margin sit beyond the Borrmann fan.
    The pad carries only the decayed tail of the fan, so it can be
    trimmed when a wide fan at an off-nominal energy would push the
    kernel past the device's shared-memory ceiling.
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
        and `shift_px` is the per-step pixel shift, snapped to an
        integer when it is within round-off of one.
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
