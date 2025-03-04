# xpartition

[![Build Status](https://github.com/spencerkclark/xpartition/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/spencerkclark/xpartition/actions)
[![codecov](https://codecov.io/gh/spencerkclark/xpartition/branch/main/graph/badge.svg?token=H1DBBSTQ2V)](https://codecov.io/gh/spencerkclark/xpartition)
[![PyPI](https://img.shields.io/pypi/v/xpartition.svg)](https://pypi.python.org/pypi/xpartition/)

This is a tool that can make writing large xarray datasets to cohesive zarr
stores from completely independent processes easier.

## Usage

The primary use-case is something like this.  Say you have a lot of netCDF files
output from a simulation or observational dataset that you would like to stitch
together into a zarr store.  If you have a way of combining those files lazily —
i.e. opening them into dask-backed arrays — into a single dataset with maybe
some additional computations, then you can write contiguous "partitions" of that
dataset out via independent processes.  A "partition" corresponds to a
contiguous group of dask "chunks." I.e. it can correspond to one or more chunks
across any number of dimensions.  A key detail is no partition straddles any
dask chunks; this makes writing from independent processes completely safe.

`xpartition` provides an accessor called `partition` that implements
`initialize_store` and `write` methods.  The pattern is to have some code that
constructs the dataset lazily, then call `initialize_store`, and finally in a
set of separate processes, call `write`.

### Simple serial example

Before illustrating a use-case of `xpartition` on a cluster, we can start with a
simple serial example.  From this example it should be straightforward to
imagine how to extend this to various distributed computing platforms, whether
HPC or cloud-based, to do the same thing in parallel.

Assume through some external package we have a function that can construct a
dataset lazily.  To incrementally write it to zarr using `xpartition` we would
only need to do the following:

```python
import xpartition

from external import construct_lazy_dataset

store = "store.zarr"
partitions = 16
partition_dims = ["tile", "time"]

ds = construct_lazy_dataset()
ds.partition.initialize_store(store)
for partition in range(partitions):
    ds.partition.write(store, partitions, partition_dims, partition)
```

`partition_dims` describes the dimensions over which to partition the dataset;
if chunks exist along dimensions that are not among the partition dimensions,
then they will all be grouped together.  If you are not particular about this,
simply using `ds.dims` will also work out of the box.

### Parallelization using `multiprocessing`

A parallel example can easily be illustrated using the built-in
`multiprocessing` library; something similar could be done with `dask.bag`:

```python
import xpartition

from external import construct_lazy_dataset

store = "store.zarr"
partitions = 16
partition_dims = ["tile", "time"]

ds = construct_lazy_dataset()
ds.partition.initialize_store(store)
with multiprocessing.get_context("spawn").Pool(partitions) as pool:
    pool.map(
        ds.partition.mappable_write(store, partitions, partition_dims),
        range(partitions)
    )
```

### Parallelization using a SLURM array job

Finally, the example below describes how one might use `xpartition` on an HPC
cluster using a SLURM array job.  We first start by writing a couple
command-line interfaces that initialize the store and write a partition.  We'll
start with one called `initialize_store.py`:

```python
import argparse
import xpartition

from external import construct_lazy_dataset

parser = argparse.ArgumentParser(
    prog="initialize_store",
    description="initialize a zarr store for a dataset"
)
parser.add_argument("store", help="absolute path to directory to store zarr result")

args = parser.parse_args()
ds = construct_lazy_dataset()
ds.partition.initialize_store(args.store)
```

Next we'll write one called `write_partition.py`:

```python
import argparse
import xpartition

from external import construct_lazy_dataset

parser = argparse.ArgumentParser(
    prog="write_partition",
    description="write a partition of a dataset"
)
parser.add_argument("store", help="absolute path to directory to store zarr result")
parser.add_argument("ranks", type=int, help="total number of available ranks")
parser.add_argument("rank", type=int, help="rank of job")

args = parser.parse_args()

# xpartition uses these as the dimensions to partition the jobs over.
dims = ["tile", "time"]

ds = construct_lazy_dataset()
ds.partition.write(args.store, args.ranks, dims, args.rank)
```

Now we can write a couple bash scripts.  The first will be a SLURM array job
that writes all the partitions.  The second will be a "main" script that
controls the whole workflow.

We call this one `write_partition.sh`:

```
#!/bin/bash
#SBATCH --job-name=zarr-history-files-array-job
#SBATCH --output=stdout/slurm-%A.%a.out # STDOUT file
#SBATCH --error=stdout/slurm-%A.%a.err  # STDERR file
#SBATCH --time=16:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-15             # job array with index values 0, 1, 2, ... 15

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID."

STORE=$1
RANKS=16
RANK=$SLURM_ARRAY_TASK_ID

python write_partition.py $STORE $RANKS $RANK
```

And we call this one `write_zarr.sh`:

```
#!/bin/bash
set -e

STORE=$1

python initialize_store.py $STORE

# Make a local directory for the stdout and stderr of the array jobs so that
# they do not clutter up the local space.
mkdir -p stdout

# Submit the array job with the -W argument to sbatch; this tells SLURM to wait
# until all array jobs have completed before returning from this script.
sbatch -W write_partition.sh $STORE
```

Submitting the full task as is then as simple as:

```
bash write_zarr.sh /path/to/store.zarr
```

## Motivation

It is not always advantageous to let all computations be controlled by a single
dask client.  At the moment, the dask scheduler breaks down when having to
manage a large number of memory-intensive tasks, often leading to slowdowns or
out of memory errors ([this issue](https://github.com/dask/distributed/issues/6360) 
is perhaps a good summary of the state of things currently in dask). Breaking the 
problem down in the way that `xpartition` does, allows you to gain the benefits of
dask's laziness on each independent process, while working in a distributed 
environment.  *In an ideal world we wouldn't need a package like this — we would
let dask and dask distributed handle everything — but in practice that does not
work perfectly yet.*

## Installation

`xpartition` can either be installed from PyPI:

```
$ pip install xpartition
```

or directly from source:

```
$ git clone https://github.com/spencerkclark/xpartition.git
$ cd xpartition
$ pip install -e .
```

## See also

There is some overlap between what this package does and what other libraries
do, namely packages like:

- [rechunker](https://github.com/pangeo-data/rechunker)
- [pangeo-forge](https://github.com/pangeo-forge/pangeo-forge)
- [xarray-beam](https://github.com/google/xarray-beam)
