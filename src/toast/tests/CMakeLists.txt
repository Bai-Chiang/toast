
# Install the python files

install(FILES
    __init__.py
    _helpers.py
    runner.py
    mpi.py
    env.py
    dist.py
    instrument.py
    intervals.py
    rng.py
    math_misc.py
    qarray.py
    fft.py
    healpix.py
    config.py
    observation.py
    timing.py
    pixels.py
    weather.py
    noise.py
    ops_sim_satellite.py
    ops_sim_ground.py
    ops_sim_tod_noise.py
    ops_stokes_weights.py
    ops_common_mode_noise.py
    ops_sim_tod_dipole.py
    ops_sim_tod_atm.py
    ops_sim_tod_conviqt.py
    ops_sim_tod_totalconvolve.py
    ops_flag_sso.py
    ops_statistics.py
    ops_mapmaker_utils.py
    ops_mapmaker_binning.py
    ops_mapmaker_solve.py
    ops_mapmaker.py
    covariance.py
    ops_pointing_detector.py
    ops_pointing_healpix.py
    ops_pointing_wcs.py
    ops_memory_counter.py
    ops_scan_map.py
    ops_scan_healpix.py
    ops_scan_wcs.py
    ops_madam.py
    ops_gainscrambler.py
    template_periodic.py
    template_amplitudes.py
    template_hwpss.py
    template_offset.py
    template_fourier2d.py
    template_subharmonic.py
    template_gain.py
    ops_sim_gaindrifts.py
    ops_sim_crosstalk.py
    ops_polyfilter.py
    ops_groundfilter.py
    ops_hwpfilter.py
    ops_sim_cosmic_rays.py
    ops_time_constant.py
    spt3g.py
    ops_pixels_healpix.py
    ops_cadence_map.py
    ops_crosslinking.py
    ops_sss.py
    ops_demodulate.py
    ops_perturbhwp.py
    ops_filterbin.py
    io_hdf5.py
    io_compression.py
    ops_noise_estim.py
    ops_yield_cut.py
    ops_elevation_noise.py
    ops_signal_diff_noise.py
    accelerator.py
    DESTINATION ${PYTHON_SITE}/toast/tests
)
