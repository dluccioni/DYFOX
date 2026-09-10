"""Measurements on simulated exit waves and images.

    metrics   crop_slices, radial_profile, ssim_pair
    topology  winding_number, vortex_census
    oam       oam_spectrum, oam_spectrum_fft
    features  locate, features, rich_features, loo_accuracy
    rocking   planewave_rocking

Every function takes its grid pitch explicitly and works on CuPy or
NumPy input, so nothing here needs a GPU.
"""
