#! /usr/bin/env python
import time
import os
import argparse
import re
import subprocess as sub

import numpy as np



parser = argparse.ArgumentParser(description='Run single file')
parser.add_argument('--max_jobs', default=300, type=int,
                    help='number of jobs to run concurrently')
parser.add_argument('--njobs', default=10,
                    type=int, help='number of jobs total to run')
parser.add_argument('--njobs_prior', default=10,
                    type=int, help='number of jobs total to run')
parser.add_argument('--files', default=10,
                    type=int, help='The number of files')
parser.add_argument('--config_file', default='config.yaml',
                    help='config file to pass')
parser.add_argument('--n_threads', default=36, type=int,
                    help='number of thread per job')
parser.add_argument('--exe', default='simulate.py',
                    help='executable to run')
parser.add_argument('--bank', default='shear',
                    help='which bank to charge to')
parser.add_argument('--partition', default='pbatch',
                    help='which submisstion partition')
parser.add_argument('--name', default='name',
                    help='name of jobs')
parser.add_argument('--hours', type=int, default=2,
                    help='how many hours')
parser.add_argument('--mins', type=int, default=0,
                    help='how many minutes')
parser.add_argument('--start', type=int, default=0,
                    help='how many minutes')
parser.add_argument('--args', default='',
                    help='extra arguments')
parser.add_argument('--prior_args', default='',
                    help='extra arguments')
parser.add_argument('--dry_run', action='store_true',
                    help="don't submit files just generate them")
parser.add_argument('--no_prior', action='store_true',
                    help="don't run prior")
parser.add_argument('--submit_prior', action='store_true',
                    help="submit prior to batch system")
args = parser.parse_args()


if os.path.exists('logs') is False:
    os.makedirs('logs')

if os.path.exists('submit') is False:
    os.makedirs('submit')

indexes = list(range(args.start, args.start+args.files))
index_lists = [indexes[i::args.njobs] for i in range(args.njobs)]

if args.submit_prior:
    args.no_prior = True
if args.no_prior is False:
    cmd = f"python bfd_desc_sims/bin/simulate_scarlet_grid.py {args.config_file} run_prior=True run_pqr=False njobs=0 {args.prior_args}"
    if args.dry_run is False:
        print(cmd)
        pipe = sub.call(cmd, stdout=sub.PIPE, shell=True)
    else:
        print(cmd)

if args.submit_prior:
    cmd = f"python bfd_desc_sims/bin/simulate_scarlet_grid.py {args.config_file} run_prior=True run_pqr=False njobs=0 {args.prior_args}"
    output = f"logs/log.prior.{args.name}"

    submit_text = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -c {args.n_threads}
#SBATCH --output={output}
#SBATCH -t {args.hours}:{args.mins:02d}:00
#SBATCH -A {args.bank}
#SBATCH -p {args.partition}
{cmd} """

    if args.dry_run is False:
        submit_file = f"submit/submit_prior_{args.name}.cmd"
        ofile = open(submit_file, 'w')
        ofile.write(submit_text)
        ofile.close()
        pipe = sub.Popen(['sbatch', submit_file], stdout=sub.PIPE)
    else:
        print(cmd)
    index_lists = []

index = args.start
for i, index_list in enumerate(index_lists):
    time.sleep(0.2)
    n_iter = int(np.ceil(len(index_list)/args.n_threads))
    sub_lists = [index_list[i::n_iter] for i in range(n_iter)]
    cmd = ""
    for j,sub_list in enumerate(sub_lists):
        cmd += f"python bfd_desc_sims/bin/simulate_scarlet_grid.py {args.config_file} njobs={len(sub_list)} index={index} {args.args}\n"
        index += len(sub_list)
    output = f"logs/log.{i}.{args.name}"
    submit_text = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -c {args.n_threads}
#SBATCH --output={output}
#SBATCH -t {args.hours}:{args.mins:02d}:00
#SBATCH -A {args.bank}
#SBATCH -p {args.partition}
{cmd} """

    submit_file = f"submit/submit_{i}_{args.name}.cmd"
    ofile = open(submit_file, 'w')
    ofile.write(submit_text)
    ofile.close()
    if args.dry_run is False:
        pipe = sub.Popen(['sbatch', submit_file], stdout=sub.PIPE)

