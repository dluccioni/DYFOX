# DYFOX (DYnamical diffraction and Fourier Optics X-ray simulator)

Simulation code for spiral phase contrast dark-field X-ray microscopy:
dynamical diffraction of a dislocation's exit wave from a strained
crystal, and the Fourier optics that images it through a spiral phase
plate.

A dislocation's diffracted exit wave carries orbital angular momentum of
charge m = -g.b about the core. Put a spiral phase plate of charge
l = +g.b at the back focal plane of the objective and that vortex is
filled in, so one exposure fixes the sign of the Burgers vector. The
code here is what you need to simulate that, or anything else built from
the same two ingredients.

## Requirements

A CUDA GPU for the Laue solver, which is a CuPy `RawKernel` integrating
in FP64. Everything else, the Bragg solver included, is NumPy and runs
anywhere.

```
pip install -r requirements.txt      # pick the cupy-cuda wheel for your CUDA
```

Nothing to install and no package to build: put the repository root on
`sys.path` and import.

```python
import sys
sys.path.insert(0, "/path/to/SPC-DFXM")
```

## Describing an experiment

Four objects, each built explicitly and passed where it is needed.
Nothing is a module-level default, so two crystals or two precisions can
coexist in one process and neither can quietly capture the other.

```python
from crystal import DIAMOND, CrystalParams
from laue import LaueGrid
from optics import Optics

xtal = CrystalParams.from_reflection(DIAMOND, (4, 0, 0), E_keV=17.0,
                                     f_prime=0.003, f_dprime=0.002)
grid = LaueGrid(xtal, t_crystal=273e-6, dx=0.25e-6, beam_width=120e-6)
lens = Optics(NA=1e-4, f_lens=0.274, xtal=xtal)
```

`crystal.py` turns a material, a reflection and a photon energy into the
wavelength, the Bragg angle, the three susceptibility couplings and the
extinction length. `DIAMOND` ships with it; any other material is a
`Material` with your own lattice parameter, basis, atomic number,
Poisson ratio and form factors.

`python info.py --hkl 220 --energy 12.5 --gpu` prints all of it, plus
what the grid will cost you on the device.

## The two solvers

**`laue/`, transmission.** Both amplitudes go into the crystal, so this
is an initial-value march from the entrance face. The displacement phase
`H = 2*pi*g.u` is precomputed on a 3-D grid from the Volterra field of
however many dislocation segments you give it, and folded into the
off-diagonal couplings as `exp(-+iH)`, so the kernel integrates the
physical wave directly. No envelope approximation, no phase-restoration
step afterwards.

The depth step is tied to the transverse grid, `dz = dx / tan(theta_B)`,
so the Borrmann fan advances exactly one pixel per step and the
transport is an index shift. This is not a detail. Interpolating at a
fractional pixel per step, compounded over a thousand steps, acts as a
few-micrometre blur, and it costs about 65x in the peak of a
micrometre-scale weak-beam signal.

**`bragg/`, reflection.** Different in kind, not just in sign. The
diffracted beam leaves through the entrance surface, so `D_0(0)` and
`D_g(t) = 0` are a two-point boundary-value problem that cannot be
marched in one pass. It is solved by shooting on `D_g(0)` with
preconditioned GMRES, with the same exact-advection depth step. Closed
forms for a perfect crystal, semi-infinite and finite-thickness Riccati,
come with it so the grid solver can be checked against something
independent.

Supporting them: `dislocations.py` for segments and the displacement
phase they impose, `optics.py` for the imaging stage, `units.py` for
rocking angle against deviation parameter, `illumination.py` for beams
that are partially coherent because they are focused or polychromatic,
`detector.py` for what a pixel array actually records off an obliquely
viewed exit surface, `analysis/` for the measurements you take off a
simulated image, and `backend.py` for precision and the one thing that
has to know whether an array lives on a GPU.

## A dislocation, start to finish

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

# One edge dislocation on [101], b = (a/2)[10-1], crossing the beam
# column at mid-thickness.
b = burgers_vector(xtal.U_lab, [1, 0, -1], xtal.b_mag, sign=+1)
z0 = grid.t_crystal / 2
seg = straight_segment([1, 0, 1] / np.sqrt(2), b,
                       [-z0 * xtal.tan_tB, 0.0, z0], 800e-6)

s_dev = units.deviation_from_urad([0.0, 117.5], xtal)   # strong, then weak beam
waves = solve_dislocation(seg, s_dev, grid, xtal)

open_img = lens.image(waves, grid.dx, ell=0)
spiral = lens.image(waves, grid.dx, ell=+2)             # l = +g.b
print("core is", float(spiral[0].max() / open_img[0].max()), "x brighter")
```

`solve_dislocation` takes a list of segments, so a field of dislocations
is the same call with a longer list, and it takes an array of deviation
parameters, so a rocking scan is one batched launch sharing a single
displacement-phase grid. Solves are expensive and fully determined by
their inputs, so `laue.WaveCache` will keep them on disk for you.

## Conventions the solver is anchored to

These matter if you extend it, and each is checked rather than assumed.

1. **Coupling sign.** `sigma_h e^{-iH}` feeds `D_g` and
   `sigma_hbar e^{+iH}` feeds `D_0`, with `H = 2*pi*g.u`, the convention
   for amplitudes riding on `e^{+ik.r}`. Anchored by imposing a uniform
   lattice tilt `H = q*y` and requiring the exit phase to come out
   as `-q*y`.
2. **Lab frame.** Rows of `U_lab` are x, y, z in crystal coordinates,
   with z the surface normal and, for symmetric Laue, g perpendicular to
   it. `lab_frame(normal_hkl, g_hkl)` builds one. For the diamond (400)
   case that is the cube frame of a (001) plate, `U_lab = I`, det = +1.
3. **Absorption.** Atomic factors carry `-i f''`, which makes
   `Re(sigma_0) < 0`, so the crystal attenuates. `F_0` uses forward
   scattering `f(0) = Z + f'`, not `f0(q)`.
4. **`sigma_hbar`** comes from `F_{-g} = f * conj(S_g)`: the geometric
   sum conjugates, the complex atomic factor does not.
5. **Spiral pupil.** For a centred plate the DC bin is zeroed, because
   the bin-average of `e^{i l phi}` across the singular bin is zero.
   `arctan2(0, 0) = 0` would instead pass the object's mean field
   straight through.
6. **Elastic constants** belong to the material. Diamond's Poisson
   ratio here is the Voigt-Reuss-Hill value 0.069 from C11, C12 and C44,
   not the 0.28 that gets quoted for it.

## What is engine and what is not

The repository root is the engine: `crystal.py`, `laue/`, `bragg/`,
`dislocations.py`, `optics.py`, `illumination.py`, `detector.py`,
`units.py`, `analysis/`, `backend.py`. It carries no experiment. There
is no default crystal, no default energy, no default objective, and
nothing in it reads a configuration file.

`spc_dfxm/` is one application of it: the scripts that reproduce the
figures of the accompanying paper, together with the crystal, energy,
aperture and defect scene that paper chose. Nothing at the root imports
it. `examples/` is four short programs that use the engine for
something else, and `regression/` is the gate that checks a cold run of
the paper layer against a stored one, file by file.

If you are here for the simulation code, you want the root and
`examples/`. If you are here to reproduce a figure, you want
`spc_dfxm/README.md`.

## Citation

See `CITATION.cff`. Released under the MIT License.
