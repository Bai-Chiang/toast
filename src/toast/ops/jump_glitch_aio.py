# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

#import copy
import os
from pathlib import Path

import numpy as np
import traitlets
from astropy import units as u
from scipy.ndimage import median_filter
from scipy.special import erf

#from .. import qarray as qa
#from ..intervals import IntervalList
from ..mpi import MPI
#from ..noise import Noise
#from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs, List
#from ..utils import Environment, Logger, name_UID
from ..utils import Logger
from .operator import Operator

@trait_docs
class JumpGlitchDetector(Operator):
    """An operator that identifies and corrects jumps and glitches in the data"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(defaults.det_data, help="Observation detdata key to analyze")

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for detector sample flagging",
    )

    reset_det_flags = Bool(
        False,
        help="Replace existing detector flags",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
        help="Bit mask value for optional shared flagging",
    )

    view = Unicode(
        None,
        allow_none=True,
        help="Find jumps and glitches in this view",
    )

    buffer_radius = Int(
        5,
        help="Number of additional samples to flag around a jump or a glitch",
    )

    medfilt_kernel_size = Int(
        11,
        help="Median filter kernel width. A positive odd number",
    )

    nsigma_sharp_detection = Float(
        5.0,
        help="Detection threshold for sharp anomalies."
        #"Phi(-nsigma_sharp_detection) * total_sample much smaller than 1,"
        #"where Phi() is normal distribution CDF.",
    )

    nsigma_blunt_candidate = Float(
        5.0,
        help="Blunt anomalies that need further identification threshold."
        #"Phi(-nsigma_blunt_candidate) * total_sample around order 10, where"
        #"Phi() is normal distribution CDF.",
    )

    nsample_radius = List(
        [25, 500],
        help="List of number of samples around flagged region to determine"
        "glitch and jumps."
    )

    nsigma_blunt_detection = List(
        [10.0, 10.0],
        help="After finding all blunt anomalies candidates. The detection"
        "threshold for confirming anomalies is real probaly not sky signal."
        "Only if the detection is significant in both local and"
        "global will be flagged.",
    )

    jump_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value to apply at glitch positions",
    )

    glitch_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value to apply at glitch positions",
    )

    fix_glitches = Bool(
        True,
        help="Fix glitches with best effort guess",
    )

    fix_jumps = Bool(
        True,
        help="Fix jumps with best effort guess",
    )

    anomy_separation_min = Int(
        5,
        allow_none=True,
        help="Minimum number of separation between anomalies (not including"
        "`buffer_radius`). If two anomalies are within this limit, they will be"
        "merged and flagged as single anomaly.",
    )

    nanomy_limit = Int(
        50,
        help="If the detector has more than `nanom_limit` anomalies,"
        "the detector and time stream will be flagged as invalid.",
    )

    stats_dir = Unicode(
        None,
        allow_none=True,
        help="Directory for writing out jumps and glitches statistics and plots."
    )

    plot_radius = Int(
        100,
        help="Number of samples around flagged region when ploting",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("medfilt_kernel_size")
    def _check_medfilt_kernel_size(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("medfilt_kernel_size cannot be negative")
        if check > 0 and check % 2 == 0:
            raise traitlets.TraitError("medfilt_kernel_size cannot be even")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net_factors = []
        self.total_factors = []
        self.weights_in = []
        self.weights_out = []
        self.rates = []

    def _Phi(self, z):
        """
        standard normal distribution CDF
        """
        return (1 + erf(z/np.sqrt(2)))/2

    def _get_stats(self, x, nsigma=1, method='center'):
        """
        get median and ±1σ magnitude.
        """
        #standard normal distribution CDF
        stats = np.zeros(3)
        if method == 'center':
            stats[0] = np.median(x)
            q3 = np.quantile(x, self._Phi(+nsigma), axis=-1, method='lower')
            q1 = np.quantile(x, self._Phi(-nsigma), axis=-1, method='higher')
            stats[1] = (q3 - stats[0])/nsigma
            stats[2] = (stats[0] - q1)/nsigma
        elif method == 'std':
            stats[0] = np.nanmean(x)
            stats[1] = stats[2] = np.nanstd(x)
        else:
            stats[0] = np.median(x)
            q3 = np.quantile(x, self._Phi(+nsigma), axis=-1, method=method)
            q1 = np.quantile(x, self._Phi(-nsigma), axis=-1, method=method)
            stats[1] = (q3 - stats[0])/nsigma
            stats[2] = (stats[0] - q1)/nsigma
        return stats

    def _keep_good_samples(self, x, nsigma_stats=3, nsigma_limit=5):
        stats = self._get_stats(x, nsigma=nsigma_stats)
        x[self._is_significant(x, stats, nsigma_limit)] = np.nan

    def _get_good_samples(self, x, nsigma_stats=3, nsigma_limit=5):
        stats = self._get_stats(x, nsigma=nsigma_stats)
        return x[np.logical_not(self._is_significant(x, stats, nsigma_limit))]

    def _is_significant(self, x, stats, nsigma):
        """
        Flag elements of x that are significant that are not in the range
        (median - nsigma * msigma, median + nsigma * psigma)
        """
        median, psigma, msigma = stats
        return (x < median - nsigma * msigma) | (x > median + nsigma * psigma)

    def _get_significance(self, x, stats):
        """
        calculate the significance given statistics
        """
        median, psigma, msigma = stats
        if x >= median:
            if psigma == 0:
                return np.inf
            return (x - median)/psigma
        else:
            if msigma == 0:
                return np.inf
            return (median - x)/msigma

    def _flag_buffer(self, x, radius, in_place=False, add_dim=False):
        """
        If radius is int, for each element in x that is flagged (True), also
        flag surrounding `radius` elements.
        If radius is tuple (radius_l radius_r), flag left and right surrounding
        unevenly.
        Add radius_l and radius_r to the end if `add_dim` is True.
        """
        if in_place and add_dim:
            msg = "_flag_buffer() in_place and add_dim cannot both be True"
            log.error(msg)
            raise RuntimeError(msg)

        if isinstance(radius, int):
            radius_l = radius_r = radius
        else:
            assert len(radius) == 2
            radius_l = radius[0]
            radius_r = radius[1]

        if in_place:
            for i in np.flatnonzero(x):
                x[max(0,i-radius_l) : min(x.shape[-1],i+1+radius_r)] = True
        else:
            if add_dim:
                buffered = np.concatenate(
                    (
                        np.repeat([False], radius_l),
                        np.zeros_like(x),
                        np.repeat([False], radius_r)
                    )
                    )
            else:
                buffered = np.zeros_like(x)
            for i in np.flatnonzero(x):
                buffered[max(0,i-radius_l) : min(x.size,i+1+radius_r)] = True
            return buffered

    def _get_flagged_slc(self, x, include_good_edge=True):
        """
        get slices of flagged region
        """
        xcp = x.copy().astype(int)
        # Set first and last element to 0 to help detect flagged region edge
        xcp[0] = 0
        xcp[-1] = 0

        if include_good_edge:
            start = np.flatnonzero(np.diff(xcp) == 1)
            end = np.flatnonzero(np.diff(xcp) == -1) + 2
        else:
            # +1 to match flagged region
            start = np.flatnonzero(np.diff(xcp) == 1) + 1
            # If first element flagged also flag zeroth element
            start[start==1] = 0
            # +1 to match flagged region
            end = np.flatnonzero(np.diff(xcp) == -1) + 1
            # similarly also flag last element if second last is flagged
            end[end==x.size-1] = x.size

        # The number of starting point should be the same as the number of ending point.
        assert start.size == end.size

        slc = []
        for i in range(start.size):
            slc.append(slice(start[i], end[i]))
        return slc

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        if self.stats_dir is not None:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import matplotlib
            matplotlib.use('Agg')
            stats_dir = Path(self.stats_dir)
            stats_dir.mkdir(parents=True, exist_ok=True)

        for ob in data.obs:
            if not ob.is_distributed_by_detector:
                msg = "Observation data must be distributed by detector, not samples"
                log.error(msg)
                raise RuntimeError(msg)
            views = ob.intervals[self.view]
            focalplane = ob.telescope.focalplane

            local_dets = ob.select_local_detectors(flagmask=self.det_mask)
            shared_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask
            print('init')
            for name in local_dets:
                print('part 1')
                if self.stats_dir is not None:
                    detector_data = ob.telescope.focalplane.detector_data
                    det_id = detector_data[np.flatnonzero(detector_data['name'] == name)]['det_info:det_id'][0]
                    ob_start = ob.shared['times'][0]
                    ob_length = ob.shared['times'][-1] - ob.shared['times'][0]
                    plots_dir = stats_dir/'plots'/ob.name
                    plots_dir.mkdir(parents=True, exist_ok=True)

                sig = ob.detdata[self.det_data][name]
                assert sig.ndim == 1    # DEBUG: make sure sig is 1d array
                det_flags = ob.detdata[self.det_flags][name]
                if self.reset_det_flags:
                    det_flags[:] = 0
                bad = np.logical_or(
                    shared_flags != 0,
                    (det_flags & self.det_flag_mask) != 0,
                )
                for iview, view in enumerate(views):
                    nsample = view.last - view.first + 1
                    ind = slice(view.first, view.last + 1)
                    sig_view = sig[ind]   # view not copy
                    assert sig_view.size == nsample    # DEBUG: Make sure nsample is correct
                    assert sig_view.ndim == 1  # DEBUG: median_filter size arg need to change if sig_view.ndim != 1

                    #assert nsample > self.nsample_min   # TODO

                    ##############################################################
                    # TMP FIX
                    ##############################################################
                    if np.ptp(sig_view) > 100:
                        ob._detflags[name] |= self.det_mask
                        det_flags[ind] |= self.det_flag_mask
                        continue
                    ##############################################################
                    # TMP FIX
                    ##############################################################

                    # peak[i] = sig[i] - ( sig[i-1] + sig[i+1] ) / 2
                    # is good at finding glitches even with HWPSS
                    peak = np.zeros(nsample)
                    peak[1:-1] = \
                        sig_view[1:-1] - 0.5*sig_view[0:-2] - 0.5*sig_view[2:]

                    trend = median_filter(sig_view, size=self.medfilt_kernel_size, mode='mirror')
                    offset = sig_view - trend

                    # diff[i] = sig[i] - sig[i-1]
                    # Use derivative on `trend` to find jumps.
                    # Since median filter should preserve sharp feature of
                    # jumps, differentiation will be large at jump point,
                    diff_trend = np.diff(trend)

                    # Derivative at center
                    diff_trend_center = np.zeros(nsample)
                    diff_trend_center[1:-1] = (trend[2:] - trend[:-2])/2
                    diff_sig_center = np.zeros(nsample)
                    diff_sig_center[1:-1] = (sig_view[2:] - sig_view[:-2])/2

                    if nsample > 741 * 10:
                        # Find the range of ±3σ
                        # 3 sigma is better capture the range of most samples
                        # if the distribution is not normal, like bimodal
                        # distribution.
                        nsigma_range = 3
                    else:
                        # Find the range of ±1σ
                        # 741 * Φ(-3) = 1.000274
                        # There won't be enough statistics for  ±3σ quantile
                        # if sample is less than 741 * 10
                        nsigma_range = 1

                    peak_stats = self._get_stats(peak[1:-1], nsigma=nsigma_range)
                    if peak_stats[1] == peak_stats[2] == 0:
                        # peak std zero means the signal is almost a straight line
                        ob._detflags[name] |= self.det_mask
                        det_flags[ind] |= self.det_flag_mask
                        continue
                    peak[0]  = peak_stats[0]
                    peak[-1] = peak_stats[0]

                    sharp_anomy = self._is_significant(peak, peak_stats, self.nsigma_sharp_detection)

                    offset_stats = self._get_stats(offset, nsigma=nsigma_range)
                    sharp_anomy &= self._is_significant(offset, offset_stats, self.nsigma_sharp_detection)

                    diff_trend_stats = self._get_stats(diff_trend, nsigma=nsigma_range)

                    #diff_trend_center_stats = self._get_stats(diff_trend_center[1:-1], nsigma=nsigma_range)
                    #diff_trend_center[0]  = diff_trend_center_stats[0]
                    #diff_trend_center[-1] = diff_trend_center_stats[0]

                    #diff_sig_stats = self._get_stats(diff_sig, nsigma=nsigma_range)

                    #diff_sig_center_stats = self._get_stats(diff_sig_center[1:-1], nsigma=nsigma_range)
                    #diff_sig_center[0]  = diff_sig_center_stats[0]
                    #diff_sig_center[-1] = diff_sig_center_stats[0]


                    # Flag both ends of diff anomaly
                    blunt_anomy = self._flag_buffer(
                            self._is_significant(diff_trend, diff_trend_stats, self.nsigma_blunt_candidate),
                            radius=(0, 1),
                            in_place=False,
                            add_dim=True,   # Add one extra element at the end to math the size of sig_view
                        )
                    ## Flag both ends of diff anomaly
                    #blunt_anomy = self._flag_buffer(
                    #        (
                    #            self._is_significant(diff_trend, diff_trend_stats, self.nsigma_blunt_candidate)
                    #          | self._is_significant(diff_sig, diff_sig_stats, self.nsigma_blunt_candidate)
                    #        ),
                    #        radius=(0, 1),
                    #        in_place=False,
                    #        add_dim=True,   # Add one extra element at the end to math the size of sig_view
                    #    )
                    ## Flag both ends of center diff anomaly
                    #blunt_anomy |= self._flag_buffer(
                    #        (
                    #            self._is_significant(diff_trend_center, diff_trend_center_stats, self.nsigma_blunt_candidate)
                    #          | self._is_significant(diff_sig_center, diff_sig_center_stats, self.nsigma_blunt_candidate)
                    #        ),
                    #        radius=1,
                    #        in_place=False
                    #    )

                    all_anomy = sharp_anomy | blunt_anomy
                    #trend[all_anomy] = np.nan
                    #diff_trend[all_anomy[:-1]] = np.nan
                    #diff_trend_std = np.nanstd(diff_trend)
                    assert blunt_anomy.shape == sig_view.shape
                    assert all_anomy.shape == sig_view.shape

                    #anomy_region = self._flag_buffer(all_anomy, radius=self.buffer_radius, in_place=False)

                    anomy_slc = self._get_flagged_slc(all_anomy, include_good_edge=False)

                    # Find anomalies that are within anomy_separation_min
                    if self.anomy_separation_min is not None:
                        for i in range(len(anomy_slc)-1):
                            if (anomy_slc[i+1].start - anomy_slc[i].stop) <= self.anomy_separation_min:
                                all_anomy[anomy_slc[i].stop:anomy_slc[i+1].start] = True
                        anomy_slc = self._get_flagged_slc(all_anomy, include_good_edge=False)

                        for i in range(len(anomy_slc)-1):
                            assert (anomy_slc[i+1].start - anomy_slc[i].stop) > self.anomy_separation_min


                    #if name == 'sch_ufm_mv5_1707067599_2_368':
                    #    for i in anomy_slc:
                    #        print(name, 'trend:', trend[i])
                    #        print(name, 'peak:', peak[i])

                    # Classify glitches and jumps
                    jump_flags = np.zeros_like(all_anomy)
                    glitch_flags = np.zeros_like(all_anomy)

                    if self.stats_dir is not None:
                        jump_mag = []
                        jump_time = [ob.shared['times'][0],]
                        jump_slc = []
                        jump_local_start_stop = []
                        jump_sample = []
                        jump_nsigma = []
                        glitch_mag = []
                        glitch_time = [ob.shared['times'][0],]
                        glitch_slc = []
                        glitch_sample = []
                        glitch_nsigma = []
                        white_noise_sigma = np.std(offset[np.logical_not(all_anomy)])

                        # Get_bias_line
                        det_data = ob.telescope.focalplane.detector_data
                        i = np.flatnonzero(det_data['name'] == name)[0]
                        bias_line =  det_data[i]['det_info:wafer:bias_line']
                        wafer =  det_data[i]['det_info:wafer:array']
                        n_det_bias_line = np.sum(det_data['det_info:wafer:bias_line'] == bias_line)


                    print('part 2')
                    nanomy = 0
                    for i_anomy in range(len(anomy_slc)):
                        if nanomy > self.nanomy_limit:
                            # Too many anomalies flag the entire detector
                            ob._detflags[name] |= self.det_mask
                            det_flags[ind] |= self.det_flag_mask
                            break

                        slc = anomy_slc[i_anomy]
                        # `start_time` and `end_time` point to good sample
                        i_start = max(0,slc.start-1)
                        i_stop = min(nsample-1,slc.stop)
                        start_time = ob.shared['times'][i_start]
                        stop_time = ob.shared['times'][i_stop]
                        anomy_time = (start_time + stop_time)/2
                        step_height = trend[i_stop] - trend[i_start]
                        step_width = i_stop - i_start

                        if self.stats_dir is not None:
                            jump_significance = []
                            local_start_stop = []

                        is_jump = True
                        local_mean = None
                        for i_radius, radius in enumerate(self.nsample_radius):
                            if not is_jump:
                                # confirmed not jump break the loop
                                break
                            else:
                                local_start = max(0, i_start-radius)
                                local_stop = min(nsample-2, i_stop+radius)
                                assert np.allclose(diff_trend[local_start:local_stop+1], np.diff(trend[local_start:local_stop+2]))
                                local_samples = diff_trend[local_start:local_stop+1]
                                if len(local_samples) > 8000:
                                    local_good_samples = self._get_good_samples(local_samples, nsigma_stats=3, nsigma_limit=self.nsigma_blunt_candidate)
                                elif len(local_samples) > 450:
                                    local_good_samples = self._get_good_samples(local_samples, nsigma_stats=2, nsigma_limit=self.nsigma_blunt_candidate)
                                else:
                                    local_good_samples = self._get_good_samples(local_samples, nsigma_stats=1, nsigma_limit=self.nsigma_blunt_candidate)
                                if len(local_good_samples) >= step_width:
                                    local_step_height = [np.sum(local_good_samples[i:i+step_width]) for i in range(local_good_samples.size+1-step_width)]
                                    if np.sum(np.isfinite(local_step_height)) >= 2 :
                                        local_mean = np.mean(local_step_height)
                                        #if i_radius == 0:
                                        #    local_mean = np.mean(local_step_height)
                                        #elif local_mean == None:
                                        #    local_mean = np.mean(local_step_height)
                                        local_std = np.std(local_step_height)
                                        local_stats = [ local_mean, local_std, local_std ]
                                        is_jump &= self._is_significant(step_height, local_stats, self.nsigma_blunt_detection[i_radius])
                                        if self.stats_dir is not None:
                                            local_start_stop.append((local_start, local_stop+1))
                                            jump_significance.append(self._get_significance(step_height, local_stats))
                                    elif np.sum(np.isfinite(local_step_height)) > 0 :
                                        #if i_radius == 0:
                                        #    local_mean = np.mean(local_step_height)
                                        if self.stats_dir is not None:
                                            local_start_stop.append((local_start, local_stop+1))
                                            jump_significance.append(np.nan)
                                    else:
                                        if self.stats_dir is not None:
                                            local_start_stop.append((local_start, local_stop+1))
                                            jump_significance.append(np.nan)
                                else:
                                    if self.stats_dir is not None:
                                        local_start_stop.append((local_start, local_stop+1))
                                        jump_significance.append(np.nan)

                        if is_jump:
                            # Difference of trend at the end is significant enough to be a jump.
                            slc_buffered = slice(max(0,slc.start-self.buffer_radius), min(nsample,slc.stop+self.buffer_radius))
                            jump_flags[slc_buffered] = True
                            nanomy += 1
                            if self.stats_dir is not None:
                                jump_slc.append(slice(max(0,slc.start-self.plot_radius),min(nsample,slc.stop+self.plot_radius)))
                                jump_local_start_stop.append(local_start_stop)
                                jump_mag.append(step_height)
                                jump_time.append(anomy_time)
                                jump_sample.append(sig_view[slc].copy())
                                jump_nsigma.append(jump_significance)
                        else:
                            # Difference of trend at the end is not significant enough to be a jump.
                            if np.sum(sharp_anomy[slc]) > 0:
                                # There is a sharp anomaly within this region.
                                slc_buffered = slice(max(0,slc.start-self.buffer_radius), min(nsample,slc.stop+self.buffer_radius))
                                glitch_flags[slc_buffered] = True
                                nanomy += 1
                                if self.stats_dir is not None:
                                    glitch_slc.append(slice(max(0,slc.start-self.plot_radius),min(nsample,slc.stop+self.plot_radius)))
                                    glitch_time.append(anomy_time)
                                    offset_slc = sig_view[slc] - trend[slc]
                                    glitch_mag.append(offset_slc[np.argmax(np.abs(offset_slc))])
                                    glitch_sample.append(sig_view[slc].copy())
                                    peak_slc = peak[slc]
                                    peak_value = peak_slc[np.argmax(np.abs(peak_slc - peak_stats[0]))]
                                    glitch_nsigma.append(
                                        max(
                                            self._get_significance(peak_value, peak_stats),
                                            self._get_significance(glitch_mag[-1],offset_stats)
                                            )
                                        )
                            else:
                                # Not jump also no glitch, it's possible this is due to point source of transient signal
                                # on the sky.
                                pass

                    print('part 3')
                    if nanomy > self.nanomy_limit:
                        # Too many anomalies flag the entire detector
                        ob._detflags[name] |= self.det_mask
                        det_flags[ind] |= self.det_flag_mask
                        continue
                    det_flags[ind][jump_flags] |= self.jump_mask
                    det_flags[ind][glitch_flags] |= self.glitch_mask


                    print('part 4')
                    # Only wirte file if there is a jump or glitch detected
                    if self.stats_dir is not None \
                            and (len(jump_mag) != 0 or len(glitch_mag) != 0):
                        jump_interval = np.diff(jump_time)
                        glitch_interval = np.diff(glitch_time)
                        # Store jump_sample and glitch_sample as 1d array to be able to save with np.savez.
                        # Recore the length of each sample to beable to recover.
                        jump_sample_len = [len(i) for i in jump_sample]
                        glitch_sample_len = [len(i) for i in glitch_sample]
                        if len(jump_sample) > 0:
                            jump_sample = np.concatenate(jump_sample)
                        if len(glitch_sample) > 0:
                            glitch_sample = np.concatenate(glitch_sample)
                        np.savez_compressed(
                                stats_dir/'{}.{}'.format(ob.name,name),
                                jump_interval=jump_interval,
                                jump_mag=jump_mag,
                                jump_time=jump_time[1:],
                                jump_slc=jump_slc,
                                jump_local_start_stop=jump_local_start_stop,
                                jump_sample=jump_sample,
                                jump_sample_len=jump_sample_len,
                                glitch_interval=glitch_interval,
                                glitch_mag=glitch_mag,
                                glitch_time=glitch_time[1:],
                                glitch_slc=glitch_slc,
                                glitch_sample=glitch_sample,
                                glitch_sample_len=glitch_sample_len,
                                ob_start=ob_start,
                                ob_length=ob_length,
                                det_id=det_id,
                                white_noise_sigma=white_noise_sigma,
                                bias_line=bias_line,
                                wafer=np.array(wafer),
                                n_det_bias_line=n_det_bias_line,
                                )

                        print('part 5')
                        njump = jump_interval.size
                        if njump > 0:
                            # plot jumps
                            fig = plt.figure(
                                    dpi=100,
                                    figsize=(18, 9+9*njump),
                                    layout="constrained",
                                )
                            gs = gridspec.GridSpec(
                                    njump+1,
                                    1,
                                    figure=fig,
                                    height_ratios=[9]+[9]*njump,
                                )

                            # plot entire time stream and flags
                            sgs = gs[0].subgridspec(2, 1, height_ratios=[2,1], hspace=0)
                            ax_sig = fig.add_subplot(sgs[0])
                            ax_flag = fig.add_subplot(sgs[1])
                            ax_sig.plot(
                                ob.shared['times'][:],
                                sig_view,
                                '-',
                            )
                            ax_flag.plot(
                                ob.shared['times'][:],
                                det_flags[:],
                                '-',
                            )
                            ax_sig.sharex(ax_flag)
                            ax_sig.xaxis.set_tick_params(labelbottom=False)
                            ax_flag.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.3f}"))
                            ax_flag.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(locs=[self.jump_mask,self.glitch_mask,self.det_flag_mask]))

                            for i in range(njump):
                                sgs = gs[i+1].subgridspec(2, 1, height_ratios=[2,1], hspace=0)
                                ax_flag = fig.add_subplot(sgs[1])
                                ax_sig = fig.add_subplot(sgs[0])
                                slc = jump_slc[i]
                                ax_sig.plot(
                                        ob.shared['times'][slc],
                                        sig_view[slc],
                                    '-',
                                )
                                ax_sig.plot(
                                        ob.shared['times'][slc],
                                        trend[slc],
                                    '--',
                                )
                                label = 'jump mag: {:.5f}  '.format(jump_mag[i])
                                for i_radius, radius in enumerate(self.nsample_radius):
                                    label += '$\sigma_{}{}{}$={:.2f}  '.format('{', radius, '}', jump_nsigma[i][i_radius])
                                label += 't=({:.2f},{:.2f})'.format(ob.shared['times'][slc.start], ob.shared['times'][slc.stop-1])
                                ax_flag.plot(
                                    ob.shared['times'][slc],
                                    det_flags[slc],
                                    '-',
                                    label=label,
                                )
                                ax_sig.sharex(ax_flag)
                                ax_sig.xaxis.set_tick_params(labelbottom=False)
                                for i_radius, radius in enumerate(self.nsample_radius):
                                    if radius < self.plot_radius:
                                        ax_sig.axvline(x=ob.shared['times'][jump_local_start_stop[i][i_radius][0]], color='tab:gray')
                                        ax_sig.axvline(x=ob.shared['times'][jump_local_start_stop[i][i_radius][1]], color='tab:gray')
                                ax_flag.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.3f}"))
                                ax_flag.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(locs=[self.jump_mask,self.glitch_mask,self.det_flag_mask]))
                                ax_flag.legend(loc="best")

                            fig.suptitle('{}  {}  {}'.format(ob.name, name, det_id ))
                            plt.savefig(plots_dir/'{}.{}.jump.png'.format(ob.name, name))
                            plt.close(fig)


                        nglitch = glitch_interval.size
                        if nglitch > 0:
                            # plot glitches
                            fig = plt.figure(
                                    dpi=100,
                                    figsize=(18, 9+9*nglitch),
                                    layout="constrained",
                                )
                            gs = gridspec.GridSpec(
                                    nglitch+1,
                                    1,
                                    figure=fig,
                                    height_ratios=[9]+[9]*nglitch,
                                )

                            # plot entire time stream and flags
                            sgs = gs[0].subgridspec(2, 1, height_ratios=[2,1], hspace=0)
                            ax_sig = fig.add_subplot(sgs[0])
                            ax_flag = fig.add_subplot(sgs[1])
                            ax_sig.plot(
                                ob.shared['times'][:],
                                sig_view,
                                '-',
                            )
                            ax_flag.plot(
                                ob.shared['times'][:],
                                det_flags[:],
                                '-',
                            )
                            ax_sig.sharex(ax_flag)
                            ax_sig.xaxis.set_tick_params(labelbottom=False)
                            ax_flag.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.3f}"))
                            ax_flag.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(locs=[self.jump_mask,self.glitch_mask,self.det_flag_mask]))

                            for i in range(nglitch):
                                sgs = gs[i+1].subgridspec(2, 1, height_ratios=[2,1], hspace=0)
                                ax_flag = fig.add_subplot(sgs[1])
                                ax_sig = fig.add_subplot(sgs[0])
                                slc = glitch_slc[i]
                                ax_sig.plot(
                                        ob.shared['times'][slc],
                                        sig_view[slc],
                                    '-',
                                )
                                ax_sig.plot(
                                        ob.shared['times'][slc],
                                        trend[slc],
                                    '--',
                                )
                                ax_flag.plot(
                                    ob.shared['times'][slc],
                                    det_flags[slc],
                                    '-',
                                    label='glitch mag: {:.5f}  $\sigma$={:.2f}  t=({:.2f},{:.2f})'.format(glitch_mag[i], glitch_nsigma[i], ob.shared['times'][slc.start], ob.shared['times'][slc.stop-1]),
                                )
                                ax_sig.sharex(ax_flag)
                                ax_sig.xaxis.set_tick_params(labelbottom=False)
                                ax_flag.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.3f}"))
                                ax_flag.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(locs=[self.jump_mask,self.glitch_mask,self.det_flag_mask]))
                                ax_flag.legend(loc="best")

                            fig.suptitle('{}  {}  {}'.format(ob.name, name, det_id))
                            plt.savefig(plots_dir/'{}.{}.glitch.png'.format(ob.name, name))
                            plt.close(fig)


                    #if fix_glitches:
                    #    # TODO fixing

                    #if fix_jumps:

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        return prov
