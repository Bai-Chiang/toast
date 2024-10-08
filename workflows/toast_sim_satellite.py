#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple satellite simulation and makes a map.

NOTE:  This script is an example.  If you are doing a simulation for a specific
experiment, you should use a custom Focalplane class rather that the simple base class
used here.

You can see the automatically generated command line options with:

    toast_sim_satellite.py --help

Or you can dump a config file with all the default values with:

    toast_sim_satellite.py --default_toml config.toml

This script contains just comments about what is going on.  For details about all the
options for a specific Operator, see the documentation or use the help() function from
an interactive python session.

"""

import argparse
import datetime
import os
import sys
import traceback

import numpy as np
from astropy import units as u

import toast
import toast.ops
from toast.mpi import MPI


def parse_config(operators, templates, comm):
    """Parse command line arguments and load any config files.

    Return the final config, remaining args, and job size args.

    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Satellite Simulation Example.")

    # Arguments specific to this script

    parser.add_argument(
        "--focalplane", required=True, default=None, help="Input fake focalplane"
    )

    parser.add_argument(
        "--schedule", required=True, default=None, help="Input observing schedule"
    )

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="toast_sim_satellite_out",
        help="The output directory",
    )

    parser.add_argument(
        "--sample_rate",
        required=False,
        type=float,
        help="Override focalplane sampling rate [Hz]",
    )

    parser.add_argument(
        "--thinfp",
        required=False,
        type=int,
        help="Only sample the provided focalplane pixels",
    )

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.

    config, args, jobargs = toast.parse_config(
        parser,
        operators=operators,
        templates=templates,
    )

    # Create our output directory
    if comm is None or comm.rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(args.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=comm)

    return config, args, jobargs


def load_instrument_and_schedule(args, comm):
    # Load a generic focalplane file.  NOTE:  again, this is just using the
    # built-in Focalplane class.  In a workflow for a specific experiment we would
    # have a custom class.
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()

    if args.sample_rate is not None:
        sample_rate = args.sample_rate * u.Hz
    else:
        sample_rate = None
    focalplane = toast.instrument.Focalplane(
        sample_rate=sample_rate,
        thinfp=args.thinfp,
    )

    with toast.io.H5File(args.focalplane, "r", comm=comm, force_serial=True) as f:
        focalplane.load_hdf5(f.handle, comm=comm)
    log.info_rank("Loaded focalplane in", comm=comm, timer=timer)
    log.info_rank(f"Focalplane: {str(focalplane)}", comm=comm)
    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After loading focalplane:  {mem}", comm)

    # Load the schedule file
    schedule = toast.schedule.SatelliteSchedule()
    schedule.read(args.schedule, comm=comm)
    log.info_rank("Loaded schedule in", comm=comm, timer=timer)
    log.info_rank(f"Schedule: {str(schedule)}", comm=comm)
    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After loading focalplane:  {mem}", comm)

    # Create a telescope for the simulation.  Again, for a specific experiment we
    # would use custom classes for the site.
    site = toast.instrument.SpaceSite(schedule.site_name)
    telescope = toast.instrument.Telescope(
        schedule.telescope_name, focalplane=focalplane, site=site
    )
    return telescope, schedule


def use_full_pointing(job):
    # Are we using full pointing?  We determine this from whether the binning operator
    # used in the solve has full pointing enabled and also whether madam (which
    # requires full pointing) is enabled.
    full_pointing = False
    if toast.ops.madam.available() and job.operators.madam.enabled:
        full_pointing = True
    if job.operators.binner.full_pointing:
        full_pointing = True
    return full_pointing


def job_create(config, jobargs, telescope, schedule, comm):
    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)

    # Find the group size for this job, either from command-line overrides or
    # by estimating the data volume.
    full_pointing = use_full_pointing(job)
    group_size = toast.job_group_size(
        comm,
        jobargs,
        schedule=schedule,
        focalplane=telescope.focalplane,
        full_pointing=full_pointing,
    )
    return job, group_size, full_pointing


def simulate_data(job, toast_comm, telescope, schedule):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates
    world_comm = toast_comm.comm_world

    # Create the (initially empty) data

    data = toast.Data(comm=toast_comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Simulate the telescope pointing

    ops.sim_satellite.telescope = telescope
    ops.sim_satellite.schedule = schedule
    ops.sim_satellite.apply(data)
    log.info_rank("Simulated telescope pointing in", comm=world_comm, timer=timer)

    # Construct a "perfect" noise model just from the focalplane parameters

    ops.default_model.apply(data)
    log.info_rank("Created default noise model in", comm=world_comm, timer=timer)

    # Set up detector pointing

    ops.det_pointing.boresight = ops.sim_satellite.boresight

    # Set up the pointing.  Each pointing matrix operator requires a detector pointing
    # operator, and each binning operator requires a pointing matrix operator.
    ops.pixels.detector_pointing = ops.det_pointing
    ops.weights.detector_pointing = ops.det_pointing
    ops.weights.hwp_angle = ops.sim_satellite.hwp_angle
    ops.pixels_final.detector_pointing = ops.det_pointing

    ops.binner.pixel_pointing = ops.pixels
    ops.binner.stokes_weights = ops.weights

    # If we are not using a different pointing matrix for our final binning, then
    # use the same one as the solve.
    if not ops.pixels_final.enabled:
        ops.pixels_final = ops.pixels

    ops.binner_final.pixel_pointing = ops.pixels_final
    ops.binner_final.stokes_weights = ops.weights

    # If we are not using a different binner for our final binning, use the same one
    # as the solve.
    if not ops.binner_final.enabled:
        ops.binner_final = ops.binner

    # Simulate sky signal from a map.  We scan the sky with the "final" pointing model
    # in case that is different from the solver pointing model.

    ops.scan_map.pixel_dist = ops.binner_final.pixel_dist
    ops.scan_map.pixel_pointing = ops.pixels_final
    ops.scan_map.stokes_weights = ops.weights
    ops.scan_map.save_pointing = use_full_pointing(job)
    ops.scan_map.apply(data)
    log.info_rank("Simulated sky signal in", comm=world_comm, timer=timer)

    # Apply a time constant

    ops.convolve_time_constant.apply(data)
    log.info_rank("Convolved time constant in", comm=world_comm, timer=timer)

    # Simulate detector noise

    ops.sim_noise.noise_model = ops.default_model.noise_model
    ops.sim_noise.apply(data)
    log.info_rank("Simulated detector noise in", comm=world_comm, timer=timer)

    mem = toast.utils.memreport(msg="(whole node)", comm=world_comm, silent=True)
    log.info_rank(f"After simulating data:  {mem}", world_comm)

    # Optionally write out the data
    if ops.save_hdf5.volume is None:
        ops.save_hdf5.volume = os.path.join(args.out_dir, "data")
    ops.save_hdf5.apply(data)
    log.info_rank("Saved HDF5 data in", comm=world_comm, timer=timer)

    mem = toast.utils.memreport(msg="(whole node)", comm=world_comm, silent=True)
    log.info_rank(f"After saving data:  {mem}", world_comm)

    return data


def reduce_data(job, args, data):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates

    world_comm = data.comm.comm_world

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Deconvolve a time constant

    ops.deconvolve_time_constant.apply(data)
    log.info_rank("Deconvolved time constant in", comm=world_comm, timer=timer)

    # The map maker requires the the binning operators used for the solve and final,
    # the templates, and the noise model.

    ops.binner.noise_model = ops.default_model.noise_model
    ops.binner_final.noise_model = ops.default_model.noise_model

    ops.mapmaker.binning = ops.binner
    ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[tmpls.baselines])
    ops.mapmaker.map_binning = ops.binner_final
    ops.mapmaker.det_data = ops.sim_noise.det_data
    ops.mapmaker.output_dir = args.out_dir

    ops.mapmaker.apply(data)
    log.info_rank("Finished map-making in", comm=world_comm, timer=timer)

    # Optionally run Madam

    if toast.ops.madam.available():
        ops.madam.params = toast.ops.madam_params_from_mapmaker(ops.mapmaker)
        ops.madam.pixel_pointing = ops.pixels_final
        ops.madam.stokes_weights = ops.weights
        ops.madam.apply(data)
        log.info_rank("Finished Madam in", comm=world_comm, timer=timer)

    return


@toast.timing.function_timer
def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_satellite_sim (total)")
    timer0 = toast.timing.Timer()
    timer0.start()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    if "OMP_NUM_THREADS" in os.environ:
        nthread = os.environ["OMP_NUM_THREADS"]
    else:
        nthread = "unknown number of"
    log.info_rank(
        f"Executing workflow with {procs} MPI tasks, each with "
        f"{nthread} OpenMP threads at {datetime.datetime.now()}",
        comm,
    )

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    #
    # We can also set some default values here for the traits, including whether an
    # operator is disabled by default.

    operators = [
        toast.ops.SimSatellite(name="sim_satellite", detset_key="pixel"),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.ScanHealpixMap(name="scan_map", enabled=False),
        toast.ops.TimeConstant(
            name="convolve_time_constant", deconvolve=False, enabled=False
        ),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.PointingDetectorSimple(name="det_pointing"),
        toast.ops.PixelsHealpix(name="pixels"),
        toast.ops.StokesWeights(name="weights", mode="IQU"),
        toast.ops.TimeConstant(
            name="deconvolve_time_constant", deconvolve=True, enabled=False
        ),
        toast.ops.SaveHDF5(name="save_hdf5", enabled=False),
        toast.ops.BinMap(name="binner", pixel_dist="pix_dist"),
        toast.ops.MapMaker(name="mapmaker"),
        toast.ops.PixelsHealpix(name="pixels_final", enabled=False),
        toast.ops.BinMap(
            name="binner_final", enabled=False, pixel_dist="pix_dist_final"
        ),
    ]
    if toast.ops.madam.available():
        operators.append(toast.ops.Madam(name="madam", enabled=False))

    # Templates we want to configure from the command line or a parameter file.
    templates = [toast.templates.Offset(name="baselines")]

    # Parse options
    config, args, jobargs = parse_config(operators, templates, comm)

    # Load our instrument model and observing schedule
    telescope, schedule = load_instrument_and_schedule(args, comm)

    # Instantiate our operators and get the size of the process groups
    job, group_size, full_pointing = job_create(
        config, jobargs, telescope, schedule, comm
    )

    # Create the toast communicator
    toast_comm = toast.Comm(world=comm, groupsize=group_size)

    # Create simulated data
    data = simulate_data(job, toast_comm, telescope, schedule)

    # Reduce the data
    reduce_data(job, args, data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=toast_comm.comm_world)
    if toast_comm.world_rank == 0:
        out = os.path.join(args.out_dir, "timing")
        toast.timing.dump(alltimers, out)

    log.info_rank("Workflow completed in", comm=comm, timer=timer0)


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
