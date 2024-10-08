// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <accelerator.hpp>

#include <intervals.hpp>

#ifdef HAVE_OPENMP_TARGET
# pragma omp declare target
#endif // ifdef HAVE_OPENMP_TARGET

void build_noise_weighted_inner(
    int32_t const * pixel_index,
    int32_t const * weight_index,
    int32_t const * flag_index,
    int32_t const * data_index,
    int64_t const * global2local,
    double const * data,
    uint8_t const * det_flags,
    uint8_t const * shared_flags,
    int64_t const * pixels,
    double const * weights,
    double const * det_scale,
    double * zmap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    int64_t nnz,
    uint8_t det_mask,
    uint8_t shared_mask,
    int64_t n_pix_submap,
    bool use_shared_flags,
    bool use_det_flags
) {
    int32_t w_indx = weight_index[idet];
    int32_t p_indx = pixel_index[idet];
    int32_t f_indx = flag_index[idet];
    int32_t d_indx = data_index[idet];

    int64_t off_p = p_indx * n_samp + isamp;
    int64_t off_w = w_indx * n_samp + isamp;
    int64_t off_d = d_indx * n_samp + isamp;
    int64_t off_f = f_indx * n_samp + isamp;
    int64_t isubpix;
    int64_t off_wt;
    double scaled_data;
    int64_t local_submap;
    int64_t global_submap;
    int64_t zoff;

    uint8_t det_check = 0;
    if (use_det_flags) {
        det_check = det_flags[off_f] & det_mask;
    }
    uint8_t shared_check = 0;
    if (use_shared_flags) {
        shared_check = shared_flags[isamp] & shared_mask;
    }

    if (
        (pixels[off_p] >= 0) &&
        (det_check == 0) &&
        (shared_check == 0)
    ) {
        // Good data, accumulate
        global_submap = (int64_t)(pixels[off_p] / n_pix_submap);

        local_submap = global2local[global_submap];

        isubpix = pixels[off_p] - global_submap * n_pix_submap;
        zoff = nnz * (local_submap * n_pix_submap + isubpix);

        off_wt = nnz * off_w;

        scaled_data = data[off_d] * det_scale[idet];

        for (int64_t iweight = 0; iweight < nnz; iweight++) {
            #pragma omp atomic update
            zmap[zoff + iweight] += scaled_data * weights[off_wt + iweight];
        }
    }
    return;
}

#ifdef HAVE_OPENMP_TARGET
# pragma omp end declare target
#endif // ifdef HAVE_OPENMP_TARGET

void init_ops_mapmaker_utils(py::module & m) {
    m.def(
        "build_noise_weighted", [](
            py::buffer global2local,
            py::buffer zmap,
            py::buffer pixel_index,
            py::buffer pixels,
            py::buffer weight_index,
            py::buffer weights,
            py::buffer data_index,
            py::buffer det_data,
            py::buffer flag_index,
            py::buffer det_flags,
            py::buffer det_scale,
            uint8_t det_flag_mask,
            py::buffer intervals,
            py::buffer shared_flags,
            uint8_t shared_flag_mask,
            bool use_accel
        ) {
            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_pixel_index = extract_buffer <int32_t> (
                pixel_index, "pixel_index", 1, temp_shape, {-1}
            );
            int64_t n_det = temp_shape[0];

            int64_t * raw_pixels = extract_buffer <int64_t> (
                pixels, "pixels", 2, temp_shape, {-1, -1}
            );
            int64_t n_samp = temp_shape[1];

            int32_t * raw_weight_index = extract_buffer <int32_t> (
                weight_index, "weight_index", 1, temp_shape, {n_det}
            );

            // Handle the case of either 2 or 3 dims
            auto winfo = weights.request();
            double * raw_weights;
            int64_t nnz;
            if (winfo.ndim == 2) {
                nnz = 1;
                raw_weights = extract_buffer <double> (
                    weights, "weights", 2, temp_shape, {-1, n_samp}
                );
            } else {
                raw_weights = extract_buffer <double> (
                    weights, "weights", 3, temp_shape, {-1, n_samp, -1}
                );
                nnz = temp_shape[2];
            }

            int32_t * raw_data_index = extract_buffer <int32_t> (
                data_index, "data_index", 1, temp_shape, {n_det}
            );
            double * raw_det_data = extract_buffer <double> (
                det_data, "det_data", 2, temp_shape, {-1, n_samp}
            );
            int32_t * raw_flag_index = extract_buffer <int32_t> (
                flag_index, "flag_index", 1, temp_shape, {n_det}
            );

            double * raw_det_scale = extract_buffer <double> (
                det_scale, "det_scale", 1, temp_shape, {n_det}
            );

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            int64_t * raw_global2local = extract_buffer <int64_t> (
                global2local, "global2local", 1, temp_shape, {-1}
            );
            int64_t n_global_submap = temp_shape[0];

            double * raw_zmap = extract_buffer <double> (
                zmap, "zmap", 3, temp_shape, {-1, -1, nnz}
            );
            int64_t n_local_submap = temp_shape[0];
            int64_t n_pix_submap = temp_shape[1];

            // Optionally use flags

            bool use_shared_flags = true;
            uint8_t * raw_shared_flags = extract_buffer <uint8_t> (
                shared_flags, "flags", 1, temp_shape, {-1}
            );
            if (temp_shape[0] != n_samp) {
                raw_shared_flags = omgr.null_ptr <uint8_t> ();
                use_shared_flags = false;
            }

            bool use_det_flags = true;
            uint8_t * raw_det_flags = extract_buffer <uint8_t> (
                det_flags, "det_flags", 2, temp_shape, {-1, -1}
            );
            if (temp_shape[1] != n_samp) {
                raw_det_flags = omgr.null_ptr <uint8_t> ();
                use_det_flags = false;
            }

            if (offload) {
                #ifdef HAVE_OPENMP_TARGET

                int64_t * dev_pixels = omgr.device_ptr(raw_pixels);
                double * dev_weights = omgr.device_ptr(raw_weights);
                double * dev_det_data = omgr.device_ptr(raw_det_data);
                Interval * dev_intervals = omgr.device_ptr(raw_intervals);
                double * dev_zmap = omgr.device_ptr(raw_zmap);
                uint8_t * dev_shared_flags = omgr.device_ptr(raw_shared_flags);
                uint8_t * dev_det_flags = omgr.device_ptr(
                    raw_det_flags);

                // Calculate the maximum interval size on the CPU
                int64_t max_interval_size = 0;
                for (int64_t iview = 0; iview < n_view; iview++) {
                    int64_t interval_size = raw_intervals[iview].last -
                                            raw_intervals[iview].first + 1;
                    if (interval_size > max_interval_size) {
                        max_interval_size = interval_size;
                    }
                }

                # pragma omp target data map(          \
                to : raw_weight_index[0 : n_det],      \
                raw_pixel_index[0 : n_det],            \
                raw_flag_index[0 : n_det],             \
                raw_data_index[0 : n_det],             \
                raw_det_scale[0 : n_det],              \
                raw_global2local[0 : n_global_submap], \
                n_view,                                \
                n_det,                                 \
                n_samp,                                \
                max_interval_size,                     \
                nnz,                                   \
                n_pix_submap,                          \
                det_flag_mask,                         \
                shared_flag_mask,                      \
                use_shared_flags,                      \
                use_det_flags                          \
                )
                {
                    # pragma omp target teams distribute parallel for collapse(3) \
                    schedule(static,1)                                            \
                    is_device_ptr(                                                \
                    dev_pixels,                                                   \
                    dev_weights,                                                  \
                    dev_det_data,                                                 \
                    dev_det_flags,                                                \
                    dev_intervals,                                                \
                    dev_shared_flags,                                             \
                    dev_zmap                                                      \
                    )
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
                            for (int64_t isamp = 0; isamp < max_interval_size; isamp++) {
                                // Adjust for the actual start of the interval
                                int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                                // Check if the value is out of range for the current
                                // interval
                                if (adjusted_isamp > dev_intervals[iview].last) {
                                    continue;
                                }

                                build_noise_weighted_inner(
                                    raw_pixel_index,
                                    raw_weight_index,
                                    raw_flag_index,
                                    raw_data_index,
                                    raw_global2local,
                                    dev_det_data,
                                    dev_det_flags,
                                    dev_shared_flags,
                                    dev_pixels,
                                    dev_weights,
                                    raw_det_scale,
                                    dev_zmap,
                                    adjusted_isamp,
                                    n_samp,
                                    idet,
                                    nnz,
                                    det_flag_mask,
                                    shared_flag_mask,
                                    n_pix_submap,
                                    use_shared_flags,
                                    use_det_flags
                                );
                            }
                        }
                    }
                }

                #endif // ifdef HAVE_OPENMP_TARGET
            } else {
                #pragma omp parallel
                {
                    int num_threads = 1;
                    int my_thread = 0;
                    #ifdef _OPENMP
                    num_threads = omp_get_num_threads();
                    my_thread = omp_get_thread_num();
                    #endif

                    int64_t n_thread_pix = (int64_t)(n_pix_submap / num_threads);
                    int64_t my_first_subpix = my_thread * n_thread_pix;
                    int64_t my_last_subpix = (my_thread + 1) * n_thread_pix;
                    if (my_thread == num_threads - 1) {
                        my_last_subpix = n_pix_submap;
                    }
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
                            for (
                                int64_t isamp = raw_intervals[iview].first;
                                isamp <= raw_intervals[iview].last;
                                isamp++
                            ) {
                                int32_t w_indx = raw_weight_index[idet];
                                int32_t p_indx = raw_pixel_index[idet];
                                int32_t f_indx = raw_flag_index[idet];
                                int32_t d_indx = raw_data_index[idet];

                                int64_t off_p = p_indx * n_samp + isamp;
                                int64_t off_w = w_indx * n_samp + isamp;
                                int64_t off_d = d_indx * n_samp + isamp;
                                int64_t off_f = f_indx * n_samp + isamp;

                                uint8_t det_check = 0;
                                if (use_det_flags) {
                                    det_check = raw_det_flags[off_f] & det_flag_mask;
                                }
                                if (det_check != 0) {
                                    continue;
                                }

                                uint8_t shared_check = 0;
                                if (use_shared_flags) {
                                    shared_check = raw_shared_flags[isamp] & shared_flag_mask;
                                }
                                if (shared_check != 0) {
                                    continue;
                                }

                                if (raw_pixels[off_p] < 0) {
                                    continue;
                                }

                                int64_t global_submap = (int64_t)(raw_pixels[off_p] / n_pix_submap);

                                int64_t local_submap = raw_global2local[global_submap];

                                int64_t isubpix = raw_pixels[off_p] - global_submap * n_pix_submap;

                                if (isubpix < my_first_subpix || isubpix >= my_last_subpix) {
                                    continue;
                                }

                                int64_t zoff = nnz * (local_submap * n_pix_submap + isubpix);

                                int64_t off_wt = nnz * off_w;

                                double scaled_data = raw_det_data[off_d] * raw_det_scale[idet];

                                for (int64_t iweight = 0; iweight < nnz; iweight++) {
                                    raw_zmap[zoff + iweight] += scaled_data * raw_weights[off_wt + iweight];
                                }
                            }
                        }
                    }
                }
            }
            return;
        });
}
