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


def _convert_scalars_to_slices(indexers):
    """Convert a dict of xarray dimension-index pairs to solely use slices.

    Assumes that the index values have been validated already in
    _validate_indexers.

    Parameters
    ----------
    indexers : dict
        Dictionary mapping dimension names to integers or slices.

    Returns
    -------
    dict
    """
    result = {}
    for k, v in indexers.items():
        if isinstance(v, slice):
            result[k] = v
        else:
            if v == -1:
                result[k] = slice(v, None)
            else:
                result[k] = slice(v, v + 1)
    return result


def _validate_indexers(indexers, sizes):
    """Check that indexers for an array with given sizes are valid.

    xpartition does not support indexing the blocks with non-contiguous array
    regions, e.g. with slices that skip elements.  It also does not support
    indexing with anything other than an integer or slice along a dimension.

    Parameters
    ----------
    indexers : dict
        Dictionary mapping dimension names to possible indexers.
    sizes : dict
        Dictionary mapping dimension names to sizes of the array.

    Raises
    ------
    KeyError, IndexError, NotImplementedError, or ValueError depending on the
    context.
    """
    for k, v in indexers.items():
        if k not in sizes:
            raise KeyError(f"Dimension {k!r} is not a valid dimension.")
        elif _is_integer(v):
            if abs(v) > sizes[k] - 1:
                raise IndexError(
                    f"Index {v} is out of bounds for dimension {k!r} of length {sizes[k]}."
                )
        elif isinstance(v, slice):
            if v.step is not None and v.step != 1:
                raise NotImplementedError(
                    "xpartition does not support indexing with slices with a step size different than None or 1."
                )
        else:
            raise ValueError(f"Invalid indexer provided for dim {k!r}: {v}.")


def _convert_block_indexers_to_array_indexers(block_indexers, chunks):
    """Convert a dict of dask block indexers to array indexers.

    Parameters
    ----------
    block_indexers : dict
        Dictionary mapping dimension names to slices.  The slices
        represent slices in dask block space.
    chunks : dict
        Dictionary mapping dimension names to tuples representing
        the chunk structure of the given dimension.

    Returns
    -------
    dict
    """
    array_indexers = {}
    for dim, block_indexer in block_indexers.items():
        if block_indexer.start is None:
            start = 0
        else:
            start = sum(chunks[dim][: block_indexer.start])
        stop = sum(chunks[dim][: block_indexer.stop])
        array_indexers[dim] = slice(start, stop)
    return array_indexers


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

    def indexers(self, **block_indexers) -> Region:
        """Return a dict of array indexers that correspond to the block indexers.

        Parameters
        ----------
        **block_indexers
            Dimension-indexer pairs in dask block space.  These can be integers
            or contiguous slices.

        Returns
        -------
        dict

        Examples
        --------
        >>> import xarray as xr; import dask.array as darray; import xpartition
        >>> arr = darray.zeros((10, 20), chunks=(2, 5))
        >>> da = xr.DataArray(arr, dims=["x", "y"], name="foo")
        >>> da
        <xarray.DataArray 'foo' (x: 10, y: 20)>
        dask.array<zeros, shape=(10, 20), dtype=float64, chunksize=(2, 5), chunktype=numpy.ndarray>
        Dimensions without coordinates: x, y
        >>> da.blocks.indexers(x=2, y=3)
        {'x': slice(4, 6, None), 'y': slice(15, 20, None)}
        >>> da.blocks.indexers(x=2)
        {'x': slice(4, 6, None)}
        >>> da.blocks.indexers(x=slice(None, None))
        {'x': slice(0, 10, None)}
        >>> da.blocks.indexers(x=slice(None, 3))
        {'x': slice(0, 6, None)}
        >>> da.blocks.indexers(x=slice(3, None))
        {'x': slice(6, 10, None)}
        >>> da.blocks.indexers(x=2, y=slice(0, 2))
        {'x': slice(4, 6, None), 'y': slice(0, 10, None)}
        """
        _validate_indexers(block_indexers, self.sizes)
        block_indexers = _convert_scalars_to_slices(block_indexers)
        return _convert_block_indexers_to_array_indexers(block_indexers, self._chunks)

    def isel(self, **block_indexers) -> xr.DataArray:
        slices = self.indexers(**block_indexers)
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


def _merge_chunks(arr, override_chunks):
    chunks_to_update = {}
    for dim, sizes in override_chunks.items():
        if dim in arr.dims:
            axis = arr.get_axis_num(dim)
            chunks_to_update[axis] = sizes
    original_chunks = {axis: sizes for axis, sizes in enumerate(arr.chunks)}
    return {**original_chunks, **chunks_to_update}


def _zeros_like_dataarray(arr, override_chunks):
    if override_chunks is None:
        override_chunks = {}
    chunks = _merge_chunks(arr, override_chunks)
    return xr.apply_ufunc(
        dask.array.zeros_like, arr, kwargs=dict(chunks=chunks), dask="allowed"
    )


def zeros_like(ds: xr.Dataset, override_chunks=None):
    """Performant implementation of zeros_like.

    xr.zeros_like(ds).chunk(chunks) is very slow for datasets with many
    changes.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with dask-backed data variables.
    override_chunks : dict
        Dimension chunk-size pairs indicating any dimensions one would like to
        override the original chunk sizes along.  For any dimensions that are not
        present, zeros_like will use the chunk size along that dimension for each
        variable in the input Dataset.

    Returns
    -------
    xr.Dataset
    """
    return ds.apply(
        _zeros_like_dataarray, override_chunks=override_chunks, keep_attrs=True
    )


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

        schema = zeros_like(
            iOut.reindex(full_indexers), override_chunks=self.plan.output_chunks
        )
        schema = schema.drop_vars(dims_without_coords)
        schema.partition.initialize_store(self.path)

    def write(self, rank):
        logging.info(f"Writing {rank + 1} of {len(self.plan.input_partition)}")
        region = self.plan.input_partition[rank]
        iData = self.data.isel(region)
        iOut = self.func(iData)
        iOut.to_zarr(self.path, region=region)
        logging.info(f"Done writing {rank + 1}.")

    def __iter__(self):
        self._initialize_store()
        return iter(range(len(self.plan.input_partition)))
