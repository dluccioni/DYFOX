"""The CUDA kernels: three of them, templated on precision.

Each source is written once with DTYPE and the maths macros left
abstract, then instantiated as float or double by `build`. Writing two
copies by hand would be asking for them to drift apart.

    tt3d_fullfield   the production kernel. Takes the displacement phase
                     from a precomputed 3-D grid, so any number of
                     dislocations of any shape.
    tt3d_analytic    the same march with the phase evaluated in-kernel
                     from a closed form. Only valid for a single
                     straight line along x_lab, and kept because an
                     independent path through the same physics is worth
                     having.
    tt3d_pristine    no dislocation at all, for rocking curves.

All three march the Takagi-Taupin equations down the depth axis, one
thread per x-column, holding the two amplitudes in shared memory. The
depth step is chosen so a characteristic advances exactly one pixel per
step, which makes the transport an index shift rather than an
interpolation. See `laue.grid` for why that matters so much.
"""

from backend import cupy


def _typed(src, fname, precision):
    """Instantiate a DTYPE-templated source at the given precision."""
    if precision.is_fp32:
        src = (src.replace("DTYPE*", "float*").replace("DTYPE ", "float ")
               .replace("(DTYPE)", "(float)")
               .replace("FABS(", "fabsf(").replace("SQRT(", "sqrtf(")
               .replace("COS(", "cosf(").replace("SIN(", "sinf(")
               .replace("EXP(", "expf(").replace("FLOOR(", "floorf(")
               .replace("3.14159265358979323846", "3.14159265358979f"))
    else:
        src = (src.replace("DTYPE*", "double*").replace("DTYPE ", "double ")
               .replace("(DTYPE)", "(double)")
               .replace("FABS(", "fabs(").replace("SQRT(", "sqrt(")
               .replace("COS(", "cos(").replace("SIN(", "sin(")
               .replace("EXP(", "exp(").replace("FLOOR(", "floor("))
    return cupy().RawKernel(src, fname)


def _typed_analytic(src, precision):
    """As _typed, plus the two functions only the analytic kernel calls."""
    src = src.replace("ATAN2(", "ATAN2X(").replace("LOG(", "LOGX(")
    if precision.is_fp32:
        src = src.replace("ATAN2X(", "atan2f(").replace("LOGX(", "logf(")
    else:
        src = src.replace("ATAN2X(", "atan2(").replace("LOGX(", "log(")
    return _typed(src, "tt3d_analytic_batch", precision)


# --- Full-field batched kernel: exp(+/- iH) in the off-diagonal coupling ---
_FULLFIELD_SRC = r"""
#define PI 3.14159265358979323846
extern "C" __global__ void tt3d_fullfield_batch(
    DTYPE* Dg_re, DTYPE* Dg_im,
    DTYPE s0r, DTYPE s0i, DTYPE shr, DTYPE shi, DTYPE sbr, DTYPE sbi,
    DTYPE* s_dev_arr, int n_e,
    DTYPE dz, DTYPE shift_px, DTYPE ds, DTYPE dx_,
    int Nz, int Nx_, int Ny_,
    DTYPE sheet_hw, DTYPE fov_hy,
    DTYPE* H_grid)   /* (Ny, Nz, Nx): pre-computed 2*pi*g.u */
{
    extern __shared__ DTYPE smem[];
    int iy = blockIdx.x / n_e;
    int ie = blockIdx.x % n_e;
    int tid = threadIdx.x, nthreads = blockDim.x;
    DTYPE y_pix = (iy - Ny_/2)*dx_;
    DTYPE s_dev_val = s_dev_arr[ie];
    int out_off = ie * Ny_ * Nx_ + iy * Nx_;
    int curr = 0;
    #define NX_PAD (Nx_ + 1)
    #define SM(buf,arr,i) smem[(buf)*4*NX_PAD + (arr)*NX_PAD + (i)]

    if (FABS(y_pix) >= fov_hy) {
        for (int ix = tid; ix < Nx_; ix += nthreads) {
            Dg_re[out_off+ix] = 0.0; Dg_im[out_off+ix] = 0.0;
        }
        return;
    }

    /* D_0 = sheet-beam top hat in x, D_g = 0 */
    for (int ix = tid; ix < Nx_; ix += nthreads) {
        DTYPE x_pos = (ix - Nx_/2)*dx_;
        /* sheet_hw > 0: top-hat half-width; sheet_hw < 0: Gaussian
           amplitude profile with sigma_A = -sheet_hw (ID03-style
           condensed line focus). */
        DTYPE beam_val = (sheet_hw > (DTYPE)0.0)
            ? ((FABS(x_pos) < sheet_hw) ? 1.0 : 0.0)
            : EXP(-x_pos*x_pos/(2.0*sheet_hw*sheet_hw));
        SM(0,0,ix)=beam_val; SM(0,1,ix)=0.0f; SM(0,2,ix)=0.0f; SM(0,3,ix)=0.0f;
    }
    __syncthreads();

    DTYPE shm2 = shr*shr + shi*shi;
    DTYPE alpha = -2.0*PI*s_dev_val;      /* strain lives in exp(iH), not alpha */
    DTYPE gam = SQRT(shm2 + alpha*alpha*0.25);
    DTYPE exp_pr = EXP(s0r * ds);

    for (int k = 0; k < Nz-1; k++){
        int nx = 1 - curr;
        DTYPE gds_ = gam*ds, cg, sg;
        if(gam>1e-30){cg=COS(gds_);sg=SIN(gds_)/gam;} else{cg=1;sg=ds;}
        DTYPE P11r=cg, P11i=-alpha*0.5*sg, P22r=cg, P22i=alpha*0.5*sg;
        DTYPE pi_=(s0i+alpha*0.5)*ds;
        DTYPE epr=exp_pr*COS(pi_), epi=exp_pr*SIN(pi_);
        DTYPE M11r=epr*P11r-epi*P11i, M11i=epr*P11i+epi*P11r;
        DTYPE M22r=epr*P22r-epi*P22i, M22i=epr*P22i+epi*P22r;

        for (int ix = tid; ix < Nx_; ix += nthreads) {
            DTYPE H_val = H_grid[iy * Nz * Nx_ + k * Nx_ + ix];
            DTYPE cH = COS(H_val), sH = SIN(H_val);
            /* Deformed-crystal couplings: chi(r - u) modulates the g
               component by exp(-iH) and the -g component by exp(+iH)
               (H = 2 pi g.u, amplitudes on e^{+i k.r} carriers -- the
               convention anchored by the refraction sign of sigma_0).
               sigma_h * exp(-iH) feeds D_g; sigma_hbar * exp(+iH)
               feeds D_0.  (The script this replaces used the opposite
               signs together with an improper U_lab; the two flips
               cancel for the topological content, but neither was
               derivable.) */
            DTYPE P12r=(sbr*cH-sbi*sH)*sg, P12i=(sbi*cH+sbr*sH)*sg;
            DTYPE P21r=(shr*cH+shi*sH)*sg, P21i=(shi*cH-shr*sH)*sg;
            DTYPE M12r=epr*P12r-epi*P12i, M12i=epr*P12i+epi*P12r;
            DTYPE M21r=epr*P21r-epi*P21i, M21i=epr*P21i+epi*P21r;

            DTYPE src0=(DTYPE)ix+shift_px;
            int i0=(int)FLOOR(src0); DTYPE f0=src0-(DTYPE)i0;
            DTYPE d0r,d0i;
            if(i0>=0&&i0+1<Nx_){d0r=(1-f0)*SM(curr,0,i0)+f0*SM(curr,0,i0+1);
                d0i=(1-f0)*SM(curr,1,i0)+f0*SM(curr,1,i0+1);}
            else if(i0>=0&&i0<Nx_){d0r=(1-f0)*SM(curr,0,i0);d0i=(1-f0)*SM(curr,1,i0);}
            else{d0r=0;d0i=0;}
            DTYPE srcg=(DTYPE)ix-shift_px;
            int ig=(int)FLOOR(srcg); DTYPE fg=srcg-(DTYPE)ig;
            DTYPE dgr,dgi;
            if(ig>=0&&ig+1<Nx_){dgr=(1-fg)*SM(curr,2,ig)+fg*SM(curr,2,ig+1);
                dgi=(1-fg)*SM(curr,3,ig)+fg*SM(curr,3,ig+1);}
            else if(ig>=0&&ig<Nx_){dgr=(1-fg)*SM(curr,2,ig);dgi=(1-fg)*SM(curr,3,ig);}
            else{dgr=0;dgi=0;}
            SM(nx,0,ix)=M11r*d0r-M11i*d0i+M12r*dgr-M12i*dgi;
            SM(nx,1,ix)=M11r*d0i+M11i*d0r+M12r*dgi+M12i*dgr;
            SM(nx,2,ix)=M21r*d0r-M21i*d0i+M22r*dgr-M22i*dgi;
            SM(nx,3,ix)=M21r*d0i+M21i*d0r+M22r*dgi+M22i*dgr;
        }
        __syncthreads(); curr=nx;
    }

    for (int ix = tid; ix < Nx_; ix += nthreads) {
        Dg_re[out_off+ix]=(DTYPE)SM(curr,2,ix);
        Dg_im[out_off+ix]=(DTYPE)SM(curr,3,ix);
    }
    #undef NX_PAD
    #undef SM
}
"""

# --- Analytic-H kernel: closed-form phase for one line along x_lab ---
_ANALYTIC_SRC = r"""
#define PI 3.14159265358979323846
extern "C" __global__ void tt3d_analytic_batch(
    DTYPE* Dg_re, DTYPE* Dg_im,
    DTYPE s0r, DTYPE s0i, DTYPE shr, DTYPE shi, DTYPE sbr, DTYPE sbi,
    DTYPE* s_dev_arr, int n_e,
    DTYPE dz, DTYPE shift_px, DTYPE ds, DTYPE dx_,
    int Nz, int Nx_, int Ny_,
    DTYPE sheet_hw, DTYPE fov_hy,
    DTYPE y0, DTYPE z0,
    DTYPE e1y, DTYPE e1z, DTYPE e2y, DTYPE e2z,
    DTYPE C_atan, DTYPE ge1, DTYPE ge2,
    DTYPE b_perp, DTYPE nu, DTYPE a2)
{
    extern __shared__ DTYPE smem[];
    int iy = blockIdx.x / n_e;
    int ie = blockIdx.x % n_e;
    int tid = threadIdx.x, nthreads = blockDim.x;
    DTYPE y_pix = (iy - Ny_/2)*dx_;
    DTYPE s_dev_val = s_dev_arr[ie];
    int out_off = ie * Ny_ * Nx_ + iy * Nx_;
    int curr = 0;
    #define NX_PAD (Nx_ + 1)
    #define SM(buf,arr,i) smem[(buf)*4*NX_PAD + (arr)*NX_PAD + (i)]

    if (FABS(y_pix) >= fov_hy) {
        for (int ix = tid; ix < Nx_; ix += nthreads) {
            Dg_re[out_off+ix] = 0.0; Dg_im[out_off+ix] = 0.0;
        }
        return;
    }

    for (int ix = tid; ix < Nx_; ix += nthreads) {
        DTYPE x_pos = (ix - Nx_/2)*dx_;
        /* sheet_hw > 0: top-hat half-width; sheet_hw < 0: Gaussian
           amplitude profile with sigma_A = -sheet_hw (ID03-style
           condensed line focus). */
        DTYPE beam_val = (sheet_hw > (DTYPE)0.0)
            ? ((FABS(x_pos) < sheet_hw) ? 1.0 : 0.0)
            : EXP(-x_pos*x_pos/(2.0*sheet_hw*sheet_hw));
        SM(0,0,ix)=beam_val; SM(0,1,ix)=0.0f; SM(0,2,ix)=0.0f; SM(0,3,ix)=0.0f;
    }
    __syncthreads();

    DTYPE shm2 = shr*shr + shi*shi;
    DTYPE alpha = -2.0*PI*s_dev_val;
    DTYPE gam = SQRT(shm2 + alpha*alpha*0.25);
    DTYPE exp_pr = EXP(s0r * ds);
    DTYPE dy = y_pix - y0;
    DTYPE onu = 1.0 - nu;
    DTYPE c1 = b_perp / (2.0*PI);

    for (int k = 0; k < Nz-1; k++){
        int nx = 1 - curr;
        DTYPE gds_ = gam*ds, cg, sg;
        if(gam>1e-30){cg=COS(gds_);sg=SIN(gds_)/gam;} else{cg=1;sg=ds;}
        DTYPE P11r=cg, P11i=-alpha*0.5*sg, P22r=cg, P22i=alpha*0.5*sg;
        DTYPE pi_=(s0i+alpha*0.5)*ds;
        DTYPE epr=exp_pr*COS(pi_), epi=exp_pr*SIN(pi_);
        DTYPE M11r=epr*P11r-epi*P11i, M11i=epr*P11i+epi*P11r;
        DTYPE M22r=epr*P22r-epi*P22i, M22i=epr*P22i+epi*P22r;

        /* H(y, z_k): x-independent for a line along x_lab */
        DTYPE dzv = k*dz - z0;
        DTYPE d1 = dy*e1y + dzv*e1z;
        DTYPE d2 = dy*e2y + dzv*e2z;
        DTYPE r2 = d1*d1 + d2*d2;
        DTYPE r2c = (r2 > a2) ? r2 : a2;
        DTYPE H_val = C_atan * ATAN2(d2, d1);
        if (b_perp > 0.0) {
            DTYPE sm_ux = c1*d1*d2/(2.0*onu*r2c);
            DTYPE sm_uy = -c1*((1.0-2.0*nu)/(4.0*onu)*LOG(r2c)
                               + (d1*d1 - d2*d2)/(4.0*onu*r2c));
            H_val += 2.0*PI*(ge1*sm_ux + ge2*sm_uy);
        }
        DTYPE cH = COS(H_val), sH = SIN(H_val);
        /* sigma_h e^{-iH} feeds D_g; sigma_hbar e^{+iH} feeds D_0 */
        DTYPE P12r=(sbr*cH-sbi*sH)*sg, P12i=(sbi*cH+sbr*sH)*sg;
        DTYPE P21r=(shr*cH+shi*sH)*sg, P21i=(shi*cH-shr*sH)*sg;
        DTYPE M12r=epr*P12r-epi*P12i, M12i=epr*P12i+epi*P12r;
        DTYPE M21r=epr*P21r-epi*P21i, M21i=epr*P21i+epi*P21r;

        for (int ix = tid; ix < Nx_; ix += nthreads) {
            DTYPE src0=(DTYPE)ix+shift_px;
            int i0=(int)FLOOR(src0); DTYPE f0=src0-(DTYPE)i0;
            DTYPE d0r,d0i;
            if(i0>=0&&i0+1<Nx_){d0r=(1-f0)*SM(curr,0,i0)+f0*SM(curr,0,i0+1);
                d0i=(1-f0)*SM(curr,1,i0)+f0*SM(curr,1,i0+1);}
            else if(i0>=0&&i0<Nx_){d0r=(1-f0)*SM(curr,0,i0);d0i=(1-f0)*SM(curr,1,i0);}
            else{d0r=0;d0i=0;}
            DTYPE srcg=(DTYPE)ix-shift_px;
            int ig=(int)FLOOR(srcg); DTYPE fg=srcg-(DTYPE)ig;
            DTYPE dgr,dgi;
            if(ig>=0&&ig+1<Nx_){dgr=(1-fg)*SM(curr,2,ig)+fg*SM(curr,2,ig+1);
                dgi=(1-fg)*SM(curr,3,ig)+fg*SM(curr,3,ig+1);}
            else if(ig>=0&&ig<Nx_){dgr=(1-fg)*SM(curr,2,ig);dgi=(1-fg)*SM(curr,3,ig);}
            else{dgr=0;dgi=0;}
            SM(nx,0,ix)=M11r*d0r-M11i*d0i+M12r*dgr-M12i*dgi;
            SM(nx,1,ix)=M11r*d0i+M11i*d0r+M12r*dgi+M12i*dgr;
            SM(nx,2,ix)=M21r*d0r-M21i*d0i+M22r*dgr-M22i*dgi;
            SM(nx,3,ix)=M21r*d0i+M21i*d0r+M22r*dgi+M22i*dgr;
        }
        __syncthreads(); curr=nx;
    }

    for (int ix = tid; ix < Nx_; ix += nthreads) {
        Dg_re[out_off+ix]=(DTYPE)SM(curr,2,ix);
        Dg_im[out_off+ix]=(DTYPE)SM(curr,3,ix);
    }
    #undef NX_PAD
    #undef SM
}
"""

# --- Pristine batched kernel (rocking-curve reference) ---
_PRISTINE_SRC = r"""
#define PI 3.14159265358979323846
extern "C" __global__ void tt3d_pristine_batch(
    DTYPE* Dg_re, DTYPE* Dg_im,
    DTYPE s0r, DTYPE s0i, DTYPE shr, DTYPE shi, DTYPE sbr, DTYPE sbi,
    DTYPE* s_dev_arr, int n_e,
    DTYPE dz, DTYPE shift_px, DTYPE ds, DTYPE dx_,
    int Nz, int Nx_, int Ny_,
    DTYPE sheet_hw, DTYPE fov_hy)
{
    extern __shared__ DTYPE smem[];
    int iy = blockIdx.x / n_e;
    int ie = blockIdx.x % n_e;
    int tid = threadIdx.x, nthreads = blockDim.x;
    DTYPE y_pix = (iy - Ny_/2)*dx_;
    DTYPE s_dev_val = s_dev_arr[ie];
    int out_off = ie * Ny_ * Nx_ + iy * Nx_;
    int curr = 0;
    #define NX_PAD (Nx_ + 1)
    #define SM(buf,arr,i) smem[(buf)*4*NX_PAD + (arr)*NX_PAD + (i)]

    if (FABS(y_pix) >= fov_hy) {
        for (int ix = tid; ix < Nx_; ix += nthreads) {
            Dg_re[out_off+ix] = 0.0; Dg_im[out_off+ix] = 0.0;
        }
        return;
    }
    for (int ix = tid; ix < Nx_; ix += nthreads) {
        DTYPE x_pos = (ix - Nx_/2)*dx_;
        /* sheet_hw > 0: top-hat half-width; sheet_hw < 0: Gaussian
           amplitude profile with sigma_A = -sheet_hw (ID03-style
           condensed line focus). */
        DTYPE beam_val = (sheet_hw > (DTYPE)0.0)
            ? ((FABS(x_pos) < sheet_hw) ? 1.0 : 0.0)
            : EXP(-x_pos*x_pos/(2.0*sheet_hw*sheet_hw));
        SM(0,0,ix)=beam_val; SM(0,1,ix)=0.0f; SM(0,2,ix)=0.0f; SM(0,3,ix)=0.0f;
    }
    __syncthreads();

    DTYPE alpha = -2.0*PI*s_dev_val;
    DTYPE shm2=shr*shr+shi*shi;
    DTYPE gam=SQRT(shm2+alpha*alpha*0.25);
    DTYPE gds_=gam*ds, cg, sg;
    if(gam>1e-30){cg=COS(gds_);sg=SIN(gds_)/gam;} else{cg=1;sg=ds;}
    DTYPE P11r=cg,P11i=-alpha*0.5*sg, P22r=cg,P22i=alpha*0.5*sg;
    DTYPE P12r=sbr*sg,P12i=sbi*sg, P21r=shr*sg,P21i=shi*sg;
    DTYPE pr=s0r*ds, pi_=(s0i+alpha*0.5)*ds;
    DTYPE epr=EXP(pr)*COS(pi_), epi=EXP(pr)*SIN(pi_);
    DTYPE M11r=epr*P11r-epi*P11i,M11i=epr*P11i+epi*P11r;
    DTYPE M12r=epr*P12r-epi*P12i,M12i=epr*P12i+epi*P12r;
    DTYPE M21r=epr*P21r-epi*P21i,M21i=epr*P21i+epi*P21r;
    DTYPE M22r=epr*P22r-epi*P22i,M22i=epr*P22i+epi*P22r;

    for (int k=0; k<Nz-1; k++){
        int nx = 1 - curr;
        for (int ix = tid; ix < Nx_; ix += nthreads) {
            DTYPE src0=(DTYPE)ix+shift_px;
            int i0=(int)FLOOR(src0); DTYPE f0=src0-(DTYPE)i0;
            DTYPE d0r,d0i;
            if(i0>=0&&i0+1<Nx_){d0r=(1-f0)*SM(curr,0,i0)+f0*SM(curr,0,i0+1);
                d0i=(1-f0)*SM(curr,1,i0)+f0*SM(curr,1,i0+1);}
            else if(i0>=0&&i0<Nx_){d0r=(1-f0)*SM(curr,0,i0);d0i=(1-f0)*SM(curr,1,i0);}
            else{d0r=0;d0i=0;}
            DTYPE srcg=(DTYPE)ix-shift_px;
            int ig=(int)FLOOR(srcg); DTYPE fg=srcg-(DTYPE)ig;
            DTYPE dgr,dgi;
            if(ig>=0&&ig+1<Nx_){dgr=(1-fg)*SM(curr,2,ig)+fg*SM(curr,2,ig+1);
                dgi=(1-fg)*SM(curr,3,ig)+fg*SM(curr,3,ig+1);}
            else if(ig>=0&&ig<Nx_){dgr=(1-fg)*SM(curr,2,ig);dgi=(1-fg)*SM(curr,3,ig);}
            else{dgr=0;dgi=0;}
            SM(nx,0,ix)=M11r*d0r-M11i*d0i+M12r*dgr-M12i*dgi;
            SM(nx,1,ix)=M11r*d0i+M11i*d0r+M12r*dgi+M12i*dgr;
            SM(nx,2,ix)=M21r*d0r-M21i*d0i+M22r*dgr-M22i*dgi;
            SM(nx,3,ix)=M21r*d0i+M21i*d0r+M22r*dgi+M22i*dgr;
        }
        __syncthreads(); curr=nx;
    }
    for (int ix = tid; ix < Nx_; ix += nthreads) {
        Dg_re[out_off+ix]=(DTYPE)SM(curr,2,ix);
        Dg_im[out_off+ix]=(DTYPE)SM(curr,3,ix);
    }
    #undef NX_PAD
    #undef SM
}
"""

_ENTRY = {
    "fullfield": (_FULLFIELD_SRC, "tt3d_fullfield_batch"),
    "pristine": (_PRISTINE_SRC, "tt3d_pristine_batch"),
}


def build(name, precision):
    """Compile one kernel. Callers should go through `precision.kernel`."""
    if name == "analytic":
        return _typed_analytic(_ANALYTIC_SRC, precision)
    if name not in _ENTRY:
        raise KeyError(f"no kernel {name!r}; have "
                       f"{sorted(list(_ENTRY) + ['analytic'])}")
    src, fname = _ENTRY[name]
    return _typed(src, fname, precision)


def prepare(kernel, smem):
    """Opt in to the large dynamic shared-memory carveout when needed.

    A block holds four rows of Nx+1 complex amplitudes, so a wide grid
    in double precision runs past the 48 KB a kernel gets by default.
    Most CUDA devices will grant up to 99 KB on request.
    """
    if smem > 48 * 1024:
        cp = cupy()
        limit = cp.cuda.Device().attributes.get(
            "MaxSharedMemoryPerBlockOptin", 99 * 1024)
        if smem > limit:
            raise MemoryError(
                f"kernel needs {smem/1024:.0f} KB shared memory per block "
                f"but the device allows {limit/1024:.0f} KB; reduce the "
                f"x-grid (beam width / crystal thickness) or use FP32")
        kernel.max_dynamic_shared_size_bytes = smem
    return kernel


def shared_bytes(Nx, precision):
    """Dynamic shared memory one block needs for an Nx-wide grid."""
    return 2 * 4 * (Nx + 1) * precision.itemsize
