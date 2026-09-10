# DYFOX (DYnamical diffraction and Fourier Optics X-ray simulator)

A simulation engine for spiral phase contrast dark-field X-ray
microscopy. It computes the dynamical diffraction of an exit wave from a
strained crystal, and the Fourier optics that image that wave through a
spiral phase plate.

The diffracted exit wave of a dislocation carries orbital angular
momentum of charge m = -g.b about the core. A spiral phase plate of
charge l = +g.b placed in the back focal plane of the objective cancels
this winding and fills the vortex, so that a single exposure determines
the sign of the Burgers vector.

## Requirements

The transmission (Laue) solver is a CuPy `RawKernel` integrating in
FP64, and requires a CUDA-capable GPU. All remaining components,
including the reflection (Bragg) solver, are implemented in NumPy and
run on any platform.

```
pip install numpy scipy
pip install cupy-cuda12x        # transmission solver; match the CUDA toolkit
pip install scikit-image        # optional, for analysis.metrics.ssim_pair
```

There is no build step and no package to install. The repository root is
placed on `sys.path` and its modules imported directly.

```python
import sys
sys.path.insert(0, "/path/to/DYFOX")
```

## Specifying an experiment

An experiment is specified by four objects, each constructed explicitly
and passed to the routines that require it. The engine defines no
module-level defaults, so configurations differing in crystal, energy,
aperture or numerical precision may coexist within a single process
without interfering.

```python
from crystal import DIAMOND, CrystalParams
from laue import LaueGrid
from optics import Optics

xtal = CrystalParams.from_reflection(DIAMOND, (4, 0, 0), E_keV=17.0,
                                     f_prime=0.003, f_dprime=0.002)
grid = LaueGrid(xtal, t_crystal=273e-6, dx=0.25e-6, beam_width=120e-6)
lens = Optics(NA=1e-4, f_lens=0.274, xtal=xtal)
```

`crystal.py` maps a material, a reflection and a photon energy onto the
wavelength, the Bragg angle, the three susceptibility couplings and the
extinction length. `DIAMOND` is provided; any other material is defined
as a `Material` from its lattice parameter, basis, atomic number,
Poisson ratio and atomic form factors.

`xtal.report()` returns the derived quantities, and `print(grid)` the
sampling they imply.

## The two solvers

**Transmission (`laue/`).** Both amplitudes propagate into the crystal
and leave through the far face, so the problem is an initial-value one
and is integrated by a single march from the entrance surface. The
displacement phase `H = 2*pi*g.u` is evaluated on a three-dimensional
grid from the Volterra field of an arbitrary number of dislocation
segments, and enters the off-diagonal couplings as `exp(-+iH)`. The
kernel therefore integrates the physical wave directly: no envelope
approximation is made, and no phase restoration is required afterwards.

The depth step is fixed by the transverse sampling,
`dz = dx / tan(theta_B)`, so that a characteristic of the Borrmann fan
advances exactly one pixel per step and transport reduces to an index
shift. The constraint is quantitatively significant: interpolating at a
fractional pixel per step compounds over some 10^3 steps into an
effective low-pass filter several micrometres wide, attenuating the peak
of a micrometre-scale weak-beam signal by a factor of approximately 65.

**Reflection (`bragg/`).** Reflection is a distinct problem rather than
a change of sign. The diffracted beam leaves through the entrance
surface, so the conditions `D_0(0) = incident` and `D_g(t) = 0`
constitute a two-point boundary-value problem, which cannot be
integrated in a single pass. It is solved by shooting on `D_g(0)` with
preconditioned GMRES, using the same exact-advection depth step.
Closed-form reflectivities for a perfect crystal, semi-infinite and
finite-thickness Riccati, are provided as independent references against
which the grid solver is verified.

## Worked example

```python
import numpy as np

from crystal import DIAMOND, CrystalParams
from dislocations import burgers_vector, straight_segment
from laue import LaueGrid, solve_dislocation
from optics import Optics
import units

xtal = CrystalParams.from_reflection(DIAMOND, (4, 0, 0), 17.0,
                                     f_prime=0.003, f_dprime=0.002)
grid = LaueGrid(xtal, t_crystal=200e-6, dx=0.25e-6, beam_width=120e-6)
lens = Optics(1e-4, 0.274, xtal)

# One edge dislocation on [101], b = (a/2)[10-1], intersecting the beam
# column at mid-thickness.
b = burgers_vector(xtal.U_lab, [1, 0, -1], xtal.b_mag, sign=+1)
z0 = grid.t_crystal / 2
seg = straight_segment([1, 0, 1] / np.sqrt(2), b,
                       [-z0 * xtal.tan_tB, 0.0, z0], 800e-6)

s_dev = units.deviation_from_urad([0.0, 117.5], xtal)   # strong, then weak beam
waves = solve_dislocation(seg, s_dev, grid, xtal)

open_img = lens.image(waves, grid.dx, ell=0)
spiral = lens.image(waves, grid.dx, ell=+2)             # l = +g.b
print("core intensity ratio:", float(spiral[0].max() / open_img[0].max()))
```

`solve_dislocation` accepts a list of segments, so a dislocation field
is obtained from the same call with a longer list, and an array of
deviation parameters, so a rocking scan is executed as a single batched
launch sharing one displacement-phase grid. Solutions are expensive and
fully determined by their inputs; `laue.WaveCache` stores them on disk
and retrieves them by content.

## Conventions

The following fix the sign and frame choices used throughout. Each is
verified by the test suite rather than assumed.

1. **Coupling sign.** `sigma_h e^{-iH}` drives `D_g` and
   `sigma_hbar e^{+iH}` drives `D_0`, with `H = 2*pi*g.u`, the
   convention appropriate to amplitudes carried on `e^{+ik.r}`. It is
   verified by imposing a uniform lattice tilt `H = q*y` and requiring
   the exit phase to be `-q*y`.
2. **Lab frame.** The rows of `U_lab` are the x, y and z axes expressed
   in crystal coordinates, with z the surface normal and, for symmetric
   Laue, g perpendicular to it. `lab_frame(normal_hkl, g_hkl)`
   constructs such a frame. For diamond (400) this is the cube frame of
   a (001) plate, `U_lab = I`, det = +1.
3. **Absorption.** Atomic form factors carry `-i f''`, which renders
   `Re(sigma_0) < 0` and therefore attenuates the wave. `F_0` is
   evaluated from the forward-scattering factor `f(0) = Z + f'`, not
   from `f0(q)`.
4. **`sigma_hbar`** follows from `F_{-g} = f * conj(S_g)`: the geometric
   structure sum is conjugated, the complex atomic factor is not.
5. **Spiral pupil.** For a centred plate the DC bin is set to zero,
   since the bin-average of `e^{i l phi}` across the singular bin
   vanishes. Evaluating `arctan2(0, 0) = 0` would instead transmit the
   mean field of the object undiffracted.
6. **Elastic constants** are properties of the material. The Poisson
   ratio used for diamond is the Voigt-Reuss-Hill value 0.069, derived
   from C11, C12 and C44, rather than the value 0.28 frequently quoted
   for it.

## Repository contents

    crystal.py        materials, and the parameters a reflection at a
                      given energy implies
    units.py          conversion between rocking angle and deviation
                      parameter
    dislocations.py   segments, and the displacement phase they impose
    laue/             transmission solver: grid, kernels, cache
    bragg/            reflection solver and closed-form references
    optics.py         objective, pupils and the spiral phase plate
    illumination.py   partially coherent beams, by convergence or by
                      bandwidth
    detector.py       response of a pixel array to an obliquely viewed
                      exit surface
    analysis/         measurements performed on a simulated image
    backend.py        numerical precision, and the boundary between the
                      array libraries

The repository contains a simulation engine and no experimental
configuration: there is no default crystal, energy or objective, and no
module reads a configuration file.

## Citation

See `CITATION.cff`. Released under the MIT License.
