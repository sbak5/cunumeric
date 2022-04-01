#!/usr/bin/env python

# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import gc
import statistics

from benchmark import run_benchmark
from legate.timing import time

import cunumeric as np


def initialize(N, nnodes, ngpus, ft):
    A = [
        np.random.rand(nnodes * N * ngpus, N // 2).astype(ft),
        np.random.rand(nnodes * N * ngpus, N // 2).astype(ft),
        np.random.rand(nnodes * N * ngpus, N).astype(ft),
    ]
    return A


def total_space(N, nnodes, ngpus, ft):
    return (N * N * nnodes * ngpus * 4) * np.dtype(ft).itemsize


def run_block(N, I, nnodes, ngpus, ft):  # noqa: E741
    print(f"Problem Size:     [[{str(N),str(N)}] * 2 ] * {nnodes} * {ngpus}")
    print("Total Iterations: " + str(I))
    space = total_space(N, nnodes, ngpus, ft)
    print(
        "Total Size:       "
        + str(space / 1e6)
        + " MB"
        + ", "
        + str(space / 1e6 / (nnodes * ngpus))
    )
    A = initialize(N, nnodes, ngpus, ft)

    avg = [0] * 2
    concat_times = []
    sum_times = []
    total_times = []

    for idx in range(I):
        start = time()
        # Run for as many iterations as was requested
        B = np.concatenate(A, axis=-1)
        stop = time()

        sum_start = time()
        C = np.sum(B)
        sum_stop = time()
        del B
        del C
        gc.collect()
        concat_time = (stop - start) * 1e-3
        sum_time = (sum_stop - sum_start) * 1e-3
        concat_times.append(concat_time)
        sum_times.append(sum_time)
        total_times.append(sum_time + concat_time)
    concat_times = concat_times[2:]
    sum_times = sum_times[2:]
    total_times = total_times[2:]

    index_min = min(range(len(total_times)), key=total_times.__getitem__)
    index_max = max(range(len(total_times)), key=total_times.__getitem__)
    print(
        f"Average Concat:  {statistics.mean(concat_times)}, "
        f"{statistics.stdev(concat_times)}, "
        f"{concat_times[index_min]}, {concat_times[index_max]}  ms\n"
        f"Average Sum:     {statistics.mean(sum_times)}, "
        f"{statistics.stdev(sum_times)}, "
        f"{sum_times[index_min]}, {sum_times[index_max]}  ms\n"
        f"Average Total:     {statistics.mean(total_times)}, "
        f"{statistics.stdev(total_times)}, "
        f"{min(total_times)}, {max(total_times)}  ms\n"
    )

    return (avg[0] + avg[1]) * I


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
        dest="I",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=32,
        dest="P",
        help="number of bits of precision to use for the gemm computation "
        "(16,32,64)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 - "
        "normal execution)",
    )
    parser.add_argument(
        "-N",
        "--nnodes",
        type=int,
        default=1,
        dest="nnodes",
        help="number of nodes to run this app on",
    )

    parser.add_argument(
        "-G",
        "--ngpus",
        type=int,
        default=1,
        dest="ngpus",
        help="number of gpus to run this app on",
    )

    args = parser.parse_args()
    if args.P == 16:
        run_benchmark(
            run_block,
            args.benchmark,
            "HGEMM",
            (args.N, args.I, args.nnodes, args.ngpus, np.float16),
        )
    elif args.P == 32:
        run_benchmark(
            run_block,
            args.benchmark,
            "SGEMM",
            (args.N, args.I, args.nnodes, args.ngpus, np.float32),
        )
    elif args.P == 64:
        run_benchmark(
            run_block,
            args.benchmark,
            "DGEMM",
            (args.N, args.I, args.nnodes, args.ngpus, np.float64),
        )
    else:
        raise TypeError("Precision must be one of 16, 32, or 64")
