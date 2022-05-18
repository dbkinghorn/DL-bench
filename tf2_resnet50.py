#!/usr/bin/env python3
"""
TensorFlow ResNet50 GPU Benchmark

Linux only (WSL2 OK) ... maybe

DBK modified version of NGC TF2 tag:22.04-tf2-py3 container
ToDo: Replaced full cuda 11.6 with 11.6 runtime (reduced size from 4.5GB to 780MB) 

"""

import statistics as st
import time
import argparse
from pathlib import Path
import json
import subprocess
import os
import platform
from meta_data import meta_data
import linux_sysinfo as sysinfo

# ******************************************************************************
# Global Variables
# ******************************************************************************
TF2_PATH = Path("assets/tf2-ngc-22.04.run")
BENCHMARK_JOBS = ["resnet"]

# ******************************************************************************
# Utility Functions
# ******************************************************************************
def nv_gpuinfo():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return result.stdout.replace(",", " id:").strip().split("\n")


def print_result(result):
    print(f".\n. Result Summary ({result['name']})\n.")
    for k, v in result.items():
        num_str = f"{v:.4f}" if isinstance(v, float) else f"{v}"
        print(f"{k:18} = {num_str}")


def run_cmd_rtn_out(cmd, sys_env=None):
    """Run cmd in a with output polling to show progress and return stdout and return code"""

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=sys_env
    )
    cmd_out = ""
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            cmd_out += output
            print(output.strip())

    rtn_code = process.poll()
    return cmd_out, rtn_code


# ******************************************************************************
# resnet_run
# ******************************************************************************
def resnet_run(job, repeats=3, gpus=0, silent=False, batch_size=64, precision="fp32"):
    """Run TF2 Benchmark job"""

    job_path = Path(f"/workspace/nvidia-examples/cnn/")
    # if not job_path.exists():
    #    raise Exception(f"Benchmark {job} path does not exist")

    num_gpus = 1 if isinstance(gpus, int) else len(gpus.split(","))
    print("num_gpus:", num_gpus)

    cmd_flag = "multi_gpu" if num_gpus > 1 else "single_gpu"

    GPUS_USED = f"CUDA_VISIBLE_DEVICES={gpus}"
    print("GPUS_USED:", GPUS_USED)
    numsteps = {"resnet": "100"}[job]

    DEV_ORDER = "CUDA_DEVICE_ORDER=PCI_BUS_ID"

    resnet_flags = f"--num_iter={numsteps} --iter_unit=batch --batch_size={batch_size} --precision={precision}"

    if cmd_flag == "single_gpu":
        commandline = f"{TF2_PATH} --env {DEV_ORDER} --env {GPUS_USED} python {job_path}/{job}.py {resnet_flags}"
    else:
        commandline = f"{TF2_PATH } --env {DEV_ORDER} --env {GPUS_USED} mpiexec -np {num_gpus} python {job_path}/{job}.py {resnet_flags}"
    commandline = commandline.split()

    timings = []
    img_per_sec = []

    for i in range(repeats):
        start_time = time.perf_counter()
        cmd_out, rtn_code = run_cmd_rtn_out(commandline)
        timings.append(time.perf_counter() - start_time)
        img_per_sec.extend(
            [
                float(line.split()[-1])  # magic num -1 for img/sec
                for line in cmd_out.split("\n")
                if line.startswith("global_step:") and int(line.split()[1]) > 10
            ]
        )

    result = {
        "name": job,
        "nv_gpus_available": nv_gpuinfo(),
        "gpu_index_used": gpus,
        "commandline": " ".join(commandline),
        "min_time": round(min(timings), 4),
        "max_time": round(max(timings), 4),
        "median_time": round(st.median(timings), 4),
        "standard_deviation": round(st.stdev(timings), 4) if len(timings) > 1 else 0,
        "performance": round(st.median(img_per_sec), 5),
        "performance_unit": "img/sec",
    }
    print_result(result) if not silent else None
    return result


def run_jobs(jobs, **kwargs):
    results = []
    for job in jobs:
        if job in BENCHMARK_JOBS:
            results.append(resnet_run(job, **kwargs))
        else:
            print(f"\nError: Unknown benchmark {job}")
    return results


def init_output_dict(output_fh):
    output_dict = {}
    output_dict["meta"] = meta_data
    output_dict["specs"] = sysinfo.specs_dict()
    output_dict["results"] = []
    with open(output_fh, "w") as f:
        json.dump(output_dict, f, indent=4)
    return output_dict


def write_results(results, output_file):
    if not output_file.exists() or Path(output_file).stat().st_size == 0:
        with open(output_file, "w") as f:
            json.dump({"results": []}, f)

    try:
        with open(output_file, "r") as f:
            jason_results = json.load(f)

        jason_results["results"].extend(results)

        with open(output_file, "w") as f:
            json.dump(jason_results, f, indent=4)
    except Exception as e:
        print(f"\nError: {e} writing results to {output_file}")


# ******************************************************************************
# Main Command Line Interface
# ******************************************************************************
def main():
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-r", "--repeats", type=int, default=3)
        parser.add_argument("--silent", action="store_true", help="Don't print results")
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            default="results.json",
            help="Output file (JSON)",
        )
        # parser.add_argument(
        #    "-c", "--cores", type=int, default=None, help="Number of cores to use"
        # )
        parser.add_argument(
            "-g",
            "--gpus",
            default=0,
            help="List of NVIDIA GPU indexes example 0,1,2,3",
        )
        parser.add_argument(
            "jobs",
            nargs="*",
            default=BENCHMARK_JOBS,
            help="Jobs to run --list for list of jobs",
        )
        parser.add_argument(
            "-l", "--list", action="store_true", help="List available jobs"
        )
        parser.add_argument("--batch_size", default=64, help="Batch size")
        parser.add_argument(
            "--precision", default="fp32", help="Precision fp32 or fp16"
        )
        return parser.parse_args()

    args = get_args()

    repeats = args.repeats
    silent = args.silent
    gpus = args.gpus
    jobs = args.jobs
    list_jobs = args.list
    output_file = args.output
    batch_size = args.batch_size
    precision = args.precision

    if list_jobs:
        print(f"\nAvailable Jobs: {BENCHMARK_JOBS} Default is all of them")
        return

    # Initialize the output file dictionary (json)
    output_dict = init_output_dict(output_file)

    results = run_jobs(
        jobs,
        repeats=repeats,
        gpus=gpus,
        silent=silent,
        batch_size=batch_size,
        precision=precision,
    )
    write_results(results, output_file)


if __name__ == "__main__":
    main()

