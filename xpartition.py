import functools
import math

import dask.array
import numpy as np
import pandas as pd
import xarray as xr
import dataclasses
import logging

from typing import Callable, Dict, Hashable, List, Sequence, Tuple, Mapping


__version__ = "0.1.0"


DIMENSION_DIM = "dimension"
SLICE_BOUND_DIM = "slice_bound"
SLICE_BOUND_INDEX = pd.Index(["start", "stop"], name=SLICE_BOUND_DIM)
AUXILLARY_DIMENSIONS = [DIMENSION_DIM, SLICE_BOUND_DIM]

Region = Sequence[Mapping[Hashable, slice]]
Partition = Sequence[Region]


def _chunks_as_data_arrays(da: xr.DataArray) -> xr.Dataset:
    chunks = da.chunks
    arrays = []
    for dim, chunks in zip(da.dims, da.chunks):
        arrays.append(xr.DataArray(list(chunks), dims=[dim], name=f"{dim}_chunks"))
    return xr.merge(arrays)


def _block_starts(da: xr.DataArray) -> List[xr.DataArray]:
    chunks_as_data_arrays = _chunks_as_data_arrays(da)
    stops = _block_stops(da)
    return [stop - a for stop, a in zip(stops, chunks_as_data_arrays.values())]


def _block_stops(da: xr.DataArray) -> List[xr.DataArray]:
    chunks_as_data_arrays = _chunks_as_data_arrays(da)
    return [a.cumsum(a.dims) for a in chunks_as_data_arrays.values()]


def block_indices(da: xr.DataArray) -> xr.DataArray:
    """Generate an array of dask array block bounds for a DataArray.

    Parameters
    ----------
    da : xr.DataArray

    Returns
    -------
    xr.DataArray
    """
    starts = _block_starts(da)
    stops = _block_stops(da)
    dimension_index = pd.Index(da.dims, name=DIMENSION_DIM)
    starts = xr.concat(xr.broadcast(*starts), dim=dimension_index)
    stops = xr.concat(xr.broadcast(*stops), dim=dimension_index)
    result = xr.concat([starts, stops], dim=SLICE_BOUND_INDEX)

    # Put auxillary dimensions at the end to make it easier to select
    # contiguous slices.
    return result.transpose(*dimension_index, ...)


def block_to_slices(block: xr.DataArray) -> Dict[Hashable, slice]:
    """Convert a DataArray representing a single block to a dictionary of slices.

    Parameters
    ----------
    block : xr.DataArray

    Returns
    -------
    Dictionary mapping dimension names to slices.
    """
    slices = {}
    for dim in block.dimension:
        start = block.sel({SLICE_BOUND_DIM: "start", DIMENSION_DIM: dim}).item()
        stop = block.sel({SLICE_BOUND_DIM: "stop", DIMENSION_DIM: dim}).item()
        slices[dim.item()] = slice(start, stop)
    return slices


def merge_blocks(blocks: xr.DataArray, dims: Sequence[Hashable]):
    """Merge a DataArray of blocks into a single block.

    Parameters
    ----------
    blocks : xr.DataArray
        An input array of blocks; must have dimensions "slice_bound"
        and "dimension".
    dims : Sequence[Hashable]
        Dimensions to merge blocks along.

    Notes
    -------
    Assumes that the input blocks are contiguous.

    Returns
    -------
    xr.DataArray for a single merged block.
    """
    start = blocks.isel({dim: 0 for dim in dims}).sel({SLICE_BOUND_DIM: "start"})
    stop = blocks.isel({dim: -1 for dim in dims}).sel({SLICE_BOUND_DIM: "stop"})
    return xr.concat([start, stop], dim=SLICE_BOUND_INDEX)


def _scalars_to_slices(kwargs):
    result = {}
    for k, v in kwargs.items():
        if isinstance(v, slice):
            result[k] = v
        elif np.issubdtype(v, np.integer):
            result[k] = slice(v, v + 1)
        else:
            raise ValueError(f"Invalid indexer provided for dim {k}: {v}.")
    return result


@xr.register_dataarray_accessor("blocks")
class BlocksAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if not isinstance(self._obj.data, dask.array.Array):
            raise ValueError(
                "The blocks accessor is only valid for dask-backed arrays."
            )

    @property
    def _blocks(self) -> xr.DataArray:
        return block_indices(self._obj)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(len(c) for c in self._obj.chunks)

    @property
    def sizes(self) -> Dict[Hashable, int]:
        return {dim: size for dim, size in zip(self._obj.dims, self.shape)}

    def indexers(self, **kwargs) -> Region:
        kwargs = _scalars_to_slices(kwargs)
        blocks = self._blocks.isel(**kwargs)
        merged = merge_blocks(blocks, self._obj.dims)
        return block_to_slices(merged)

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
        return [self.indexers(ranks, rank, dims) for rank in range(ranks)]

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


def zeros_like(ds: xr.Dataset, chunks):
    return xr.zeros_like(ds).chunk(chunks)


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
