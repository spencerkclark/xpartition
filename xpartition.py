import functools
import math

import dask.array
import numpy as np
import xarray as xr
import dataclasses
import logging

from typing import Callable, Dict, Hashable, Sequence, Tuple, Mapping


__version__ = "0.1.0"


Region = Sequence[Mapping[Hashable, slice]]
Partition = Sequence[Region]


def _is_integer(value):
    """Check if a value is a Python or NumPy integer instance."""
    return isinstance(value, (int, np.integer))


def _convert_scalars_to_slices(kwargs):
    """Convert a set of xarray dimension-index pairs to solely use slices."""
    result = {}
    for k, v in kwargs.items():
        if isinstance(v, slice):
            result[k] = v
        elif _is_integer(v):
            result[k] = slice(v, v + 1)
        else:
            raise ValueError(f"Invalid indexer provided for dim {k}: {v}.")
    return result


def _validate_indexers(kwargs, sizes):
    """Check that indexers for an array with given sizes are valid."""
    for k, v in kwargs.items():
        if k not in sizes:
            raise KeyError(f"Dimension {k!r} is not a valid dimension.")
        elif _is_integer(v):
            if abs(v) > sizes[k] - 1:
                raise IndexError(
                    f"Index {v} is out of bounds for dimension {k!r} of length {sizes[k]}."
                )
        else:
            if not isinstance(v, slice):
                raise ValueError(f"Invalid indexer provided for dim {k!r}: {v}.")


def _convert_block_indexers_to_array_indexers(kwargs, chunks):
    """Convert a set of dask block indexers to array indexers."""
    slices = {}
    for dim, indexer in kwargs.items():
        start = sum(chunks[dim][: indexer.start])
        stop = sum(chunks[dim][: indexer.stop])
        slices[dim] = slice(start, stop)
    return slices


@xr.register_dataarray_accessor("blocks")
class BlocksAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if not isinstance(self._obj.data, dask.array.Array):
            raise ValueError(
                "The blocks accessor is only valid for dask-backed arrays."
            )

    @property
    def _chunks(self) -> Dict[Hashable, Tuple[int, ...]]:
        return {dim: self._obj.chunks[k] for k, dim in enumerate(self._obj.dims)}

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(len(c) for c in self._obj.chunks)

    @property
    def sizes(self) -> Dict[Hashable, int]:
        return {dim: size for dim, size in zip(self._obj.dims, self.shape)}

    def indexers(self, **kwargs) -> Region:
        _validate_indexers(kwargs, self.sizes)
        block_indexers = _convert_scalars_to_slices(kwargs)
        return _convert_block_indexers_to_array_indexers(block_indexers, self._chunks)

    def isel(self, **kwargs) -> xr.DataArray:
        slices = self.indexers(**kwargs)
        # TODO: should we squeeze out dimensions where scalars were passed?
        return self._obj.isel(slices)


def _write_partition_dataarray(
    da: xr.DataArray, store: str, ranks: int, dims: Sequence[Hashable], rank: int
):
    ds = da.to_dataset()
    partition = da.partition.indexers(ranks, rank, dims)
    if partition is not None:
        ds.isel(partition).to_zarr(store, region=partition)


def _write_partition_dataset(
    ds: xr.Dataset, store: str, ranks: int, dims: Sequence[Hashable], rank: int
):
    for da in ds.data_vars.values():
        if isinstance(da.data, dask.array.Array):
            partition_dims = [dim for dim in dims if dim in da.dims]
            da.partition.write(store, ranks, partition_dims, rank)


class Map(Sequence):
    """Lazy sequence"""

    def __init__(self, func, seq):
        self.seq = seq
        self.func = func

    def __getitem__(self, i):
        return self.func(self.seq[i])

    def __len__(self):
        return len(self.seq)


@xr.register_dataarray_accessor("partition")
class PartitionDataArrayAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if not isinstance(self._obj.data, dask.array.Array):
            raise ValueError(
                "The partition accessor is only valid for dask-backed arrays."
            )

    def _meta_array(self, chunks: Dict[Hashable, int]) -> xr.DataArray:
        dummy_data = dask.array.zeros(self._obj.blocks.shape)
        da = xr.DataArray(dummy_data, dims=self._obj.dims, name="blocks")
        return da.chunk(chunks)

    def _optimal_meta_chunk_sizes(
        self, ranks: int, dims: Sequence[Hashable]
    ) -> Dict[Hashable, int]:
        """Determine the optimal meta chunk sizes for the DataArray.

        Partitions are prioritized based on the ordering of the dims
        provided.  Priority means we will first make the meta chunk
        size one along those dimensions before moving to larger meta
        chunk sizes.

        Parameters
        ----------
        ranks : int
            Total number of ranks available to partition across.
        dims : Sequence[Hashable]
            Dimensions to partition among; if a dimension is left out
            no partitions will be made along that dimension.

        Returns
        -------
        Dict[Hashable, int]
        """
        chunk_sizes = {}
        for dim in dims:
            block_sizes = []
            for d, s in chunk_sizes.items():
                block_size = math.ceil(self._obj.blocks.sizes[d] / s)
                block_sizes.append(block_size)
            blocks = np.product(block_sizes)
            size = math.ceil(self._obj.blocks.sizes[dim] / (ranks // blocks))
            chunk_sizes[dim] = min(size, self._obj.blocks.sizes[dim])
        return chunk_sizes

    def partition(self, ranks, dims) -> Partition:
        """Compute a ranks-sized partition respecting dask block boundaries

        Parameters
        ----------
        ranks : int
            Total number of ranks available to partition across.
        dims : Sequence[Hashable]
            Dimensions to partition among; if a dimension is left out
            no partitions will be made along that dimension.

        Returns
        -------
        a list of disjoint regions whose union is the full coordinate space
        """
        return Map(functools.partial(self._indexers, ranks, dims), list(range(ranks)))

    def _indexers(self, ranks, dims, rank):
        """Needed for creating a partial function within the partition method."""
        return self.indexers(ranks, rank, dims)

    def indexers(self, ranks: int, rank: int, dims: Sequence[Hashable]) -> Region:
        """Partition the dask blocks across the given dims.

        Parameters
        ----------
        ranks : int
            Total number of ranks available to partition across.
        rank : int
            Specific rank to obtain the indexers for.
        dims : Sequence[Hashable]
            Dimensions to partition among; if a dimension is left out
            no partitions will be made along that dimension.

        Returns
        -------
        Dict[Hashable, slice]
        """
        if (rank - 1) > ranks:
            raise ValueError(f"Rank {rank} is greater than available ranks {ranks}.")

        meta_chunk_sizes = self._optimal_meta_chunk_sizes(ranks, dims)
        meta_array = self._meta_array(meta_chunk_sizes)
        try:
            meta_indices = np.unravel_index(rank, meta_array.blocks.shape)
        except ValueError:
            return None
        else:
            meta_indexers = dict(zip(meta_array.dims, meta_indices))
            dask_indexers = meta_array.blocks.indexers(**meta_indexers)
            return self._obj.blocks.indexers(**dask_indexers)

    def write(self, store: str, ranks: int, dims: Sequence[Hashable], rank: int):
        _write_partition_dataarray(self._obj, store, ranks, dims, rank)

    def mappable_write(
        self, store: str, ranks: int, dims: Sequence[Hashable]
    ) -> Callable[[int], None]:
        return functools.partial(
            _write_partition_dataarray, self._obj, store, ranks, dims
        )

    @property
    def _chunks(self):
        return {dim: self._obj.chunks[k] for k, dim in enumerate(self._obj.dims)}

    def map(
        self, store: str, ranks: int, dims: Sequence[Hashable], func, data
    ) -> "PartitionMapper":
        plan = _ValidWorkPlan(self, ranks, dims)
        return PartitionMapper(plan, func, data, store)


@xr.register_dataset_accessor("partition")
class PartitionDatasetAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def initialize_store(self, store: str):
        self._obj.to_zarr(store, compute=False)

    def write(self, store: str, ranks: int, dims: Sequence[Hashable], rank: int):
        _write_partition_dataset(self._obj, store, ranks, dims, rank)

    def mappable_write(
        self, store: str, ranks: int, dims: Sequence[Hashable]
    ) -> Callable[[int], None]:
        return functools.partial(
            _write_partition_dataset, self._obj, store, ranks, dims
        )


def zeros_like_dataarray(arr, chunks):
    dask_chunks = {
        arr.get_axis_num(dim): size for dim, size in chunks.items() if dim in arr.dims
    }
    return xr.apply_ufunc(
        dask.array.zeros_like, arr, kwargs=dict(chunks=dask_chunks), dask="allowed"
    )


def zeros_like(ds: xr.Dataset, chunks):
    """Performant implementation of zeros_like with a given chunk size

    xr.zeros_like(ds).chunk(chunks) is very slow for datasets with many
    changes.
    """
    return ds.apply(zeros_like_dataarray, chunks=chunks, keep_attrs=True)


class _ValidWorkPlan:
    """A mapping between input and output partitionings that will
    avoid race conditions in parallel jobs
    """

    def __init__(self, partitioner, ranks: int, dims: Sequence[Hashable]):

        self._partitioner = partitioner
        self._ranks = ranks
        self.dims = dims

    @property
    def output_chunks(self):
        return {dim: self._partitioner._chunks[dim] for dim in self.dims}

    @property
    def input_partition(self):
        return self._partitioner.partition(self._ranks, self.dims)


@dataclasses.dataclass
class PartitionMapper:
    """Evaluate a function on each region of a partition and store the output
    to a zarr store
    """

    plan: _ValidWorkPlan
    func: Callable[[xr.Dataset], xr.Dataset]
    data: xr.Dataset
    path: str

    @property
    def dims(self):
        return self.plan.dims

    def _initialize_store(self):
        region = self.plan.input_partition[0]
        iData = self.data.isel(region)
        iOut = self.func(iData)
        full_indexers = {dim: self.data[dim] for dim in self.dims}

        dims_without_coords = (set(iOut.dims) - set(iOut.indexes)) & set(self.dims)
        for dim in dims_without_coords:
            iOut = iOut.assign_coords({dim: iOut[dim]})

        schema = zeros_like(iOut.reindex(full_indexers), chunks=self.plan.output_chunks)
        schema = schema.drop_vars(dims_without_coords)
        schema.partition.initialize_store(self.path)

    def write(self, rank):
        logging.info(f"Writing {rank + 1} of {len(self.plan.input_partition)}")
        region = self.plan.input_partition[rank]
        iData = self.data.isel(region)
        iOut = self.func(iData)
        iOut.to_zarr(self.path, region=region)

    def __iter__(self):
        self._initialize_store()
        return iter(range(len(self.plan.input_partition)))
