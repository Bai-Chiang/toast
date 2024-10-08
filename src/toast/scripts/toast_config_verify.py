#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This does some simple tests of the TOAST runtime environment.
"""

import argparse
import re
import sys
import traceback
from collections import OrderedDict

import toast
from toast.config import dump_json, dump_toml, dump_yaml, load_config
from toast.mpi import Comm, get_world
from toast.utils import Environment, Logger, import_from_name, object_fullname


def main():
    env = Environment.get()
    log = Logger.get()

    mpiworld, procs, rank = get_world()
    if rank != 0:
        return

    # Split up the commandline into arguments we want to parse versus things
    # we will pass to the toast parser.
    opts = list(sys.argv[1:])

    user_opts = list()
    conf_opts = list()
    iop = 0
    in_class = False
    in_conf = False
    while iop < len(opts):
        if opts[iop] == "--out_toml":
            user_opts.append(opts[iop])
            iop += 1
            user_opts.append(opts[iop])
            iop += 1
        elif opts[iop] == "--out_json":
            user_opts.append(opts[iop])
            iop += 1
            user_opts.append(opts[iop])
            iop += 1
        elif opts[iop] == "--out_yaml":
            user_opts.append(opts[iop])
            iop += 1
            user_opts.append(opts[iop])
            iop += 1
        elif opts[iop] == "--class":
            user_opts.append(opts[iop])
            iop += 1
            in_class = True
        elif opts[iop] == "--config":
            # Config files are needed by both parsers
            user_opts.append(opts[iop])
            conf_opts.append(opts[iop])
            iop += 1
            in_conf = True
        elif in_class:
            user_opts.append(opts[iop])
            iop += 1
            if iop < len(opts) and re.match(r"^--.*", opts[iop]) is not None:
                in_class = False
        elif in_conf:
            # Config files are needed by both parsers
            user_opts.append(opts[iop])
            conf_opts.append(opts[iop])
            iop += 1
            if iop < len(opts) and re.match(r"^--.*", opts[iop]) is not None:
                in_conf = False
        else:
            conf_opts.append(opts[iop])
            iop += 1

    help = """Generate or verify config files.

    This utility can dump out default config files for specified TraitConfig
    derived classes, as well as parse input config files and commandline arguments
    to check the final configuration that results.
    """

    parser = argparse.ArgumentParser(description=help)
    parser.add_argument(
        "--out_toml",
        required=False,
        default=None,
        type=str,
        help="The output TOML config file to write",
    )
    parser.add_argument(
        "--out_yaml",
        required=False,
        default=None,
        type=str,
        help="The output YAML config file to write",
    )
    parser.add_argument(
        "--out_json",
        required=False,
        default=None,
        type=str,
        help="The output JSON config file to write",
    )
    parser.add_argument(
        "--class",
        type=str,
        required=False,
        nargs="+",
        dest="classes",
        help="One or more class names to import.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        nargs="+",
        help="One or more input config files.",
    )

    user_args, _ = parser.parse_known_args()

    # Now merge our class list with classes mentioned in the config files

    operators = dict()
    templates = dict()
    if user_args.classes is not None:
        for cname in user_args.classes:
            cls = import_from_name(cname)
            obj = cls()
            if isinstance(obj, toast.ops.Operator):
                if cname not in operators:
                    operators[cname] = obj
            elif isinstance(obj, toast.templates.Template):
                if cname not in templates:
                    templates[cname] = obj
    if user_args.config is not None:
        for conf_file in user_args.config:
            conf = load_config(conf_file)
            if "operators" in conf:
                for op_name, op_props in conf["operators"].items():
                    op_cls = op_props["class"]
                    cls = import_from_name(op_cls)
                    if op_name not in operators:
                        operators[op_name] = cls(name=op_name)
            if "templates" in conf:
                for tname, tprops in conf["templates"].items():
                    tcls = tprops["class"]
                    cls = import_from_name(tcls)
                    if tname not in templates:
                        templates[tname] = cls(name=tname)

    # Now pass this through the normal code path
    conf_parser = argparse.ArgumentParser(description="Config parsing")
    config, args, jobargs = toast.parse_config(
        conf_parser,
        operators=[y for x, y in operators.items()],
        templates=[y for x, y in templates.items()],
        opts=conf_opts,
    )

    # Instantiate everything and then convert back to a config for dumping.
    # This will automatically prune stale traits, etc.
    run = toast.create_from_config(config)
    run_vars = vars(run)

    out_config = OrderedDict()
    for sect_key, sect_val in run_vars.items():
        sect_vars = vars(sect_val)
        obj_list = list()
        for obj_name, obj in sect_vars.items():
            obj_list.append(obj)
        out_config.update(toast.config.build_config(obj_list))

    # Write the final config out
    if user_args.out_toml is not None:
        dump_toml(user_args.out_toml, out_config, comm=mpiworld)
    if user_args.out_json is not None:
        dump_json(user_args.out_json, out_config, comm=mpiworld)
    if user_args.out_yaml is not None:
        dump_yaml(user_args.out_yaml, out_config, comm=mpiworld)

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
