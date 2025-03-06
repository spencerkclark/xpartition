import collections
import dataclasses
import functools
import logging
import math
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import dask.array
import numpy as np
import xarray as xr

from xpartition.xarray_utils import get_chunks_encoding

__version__ = "2025.03.0"


Region = Union[None, Mapping[Hashable, slice]]
Partition = Sequence[Region]
HashableSlice = Tuple[Union[None, int], Union[None, int], Union[None, int]]
HashableIndexers = Union[None, Tuple[Tuple[Hashable, HashableSlice], ...]]


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
    ds = da.drop_vars(da.coords).to_dataset()
    partition = da.partition.indexers(ranks, rank, dims)
    if partition is not None:
        ds.isel(partition).to_zarr(store, region=partition)


def freeze_indexers(indexers: Region) -> HashableIndexers:
    """Return an immutable (hashable) version of the indexers."""
    if indexers is None:
        return indexers
    else:
        immutable = ((k, (s.start, s.stop, s.step)) for k, s in indexers.items())
        return tuple(sorted(immutable, key=lambda x: x[0]))


def unfreeze_indexers(frozen_indexers: HashableIndexers) -> Region:
    """Convert an immutable version of the indexers back to its usual type."""
    if frozen_indexers is None:
        return frozen_indexers
    else:
        return {k: slice(*s) for k, s in frozen_indexers}


def _collect_by_partition(
    ds: xr.Dataset, ranks: int, dims: Sequence[Hashable], rank: int
) -> Sequence[Tuple[Region, xr.Dataset]]:
    """Return a list of pairs of partitions and Datasets containing
    DataArrays that can be written out to those partitions.
    """
    dataarrays = collections.defaultdict(list)
    for da in {**ds.coords, **ds.data_vars}.values():
        if isinstance(da.data, dask.array.Array):
            partition_dims = [dim for dim in dims if dim in da.dims]
            indexers = da.partition.indexers(ranks, rank, partition_dims)
            dataarrays[freeze_indexers(indexers)].append(da.drop_vars(da.coords))
    return [(unfreeze_indexers(k), xr.merge(v)) for k, v in dataarrays.items()]


def _write_partition_dataset_via_individual_variables(
    ds: xr.Dataset, store: str, ranks: int, dims: Sequence[Hashable], rank: int
):
    for da in {**ds.coords, **ds.data_vars}.values():
        if isinstance(da.data, dask.array.Array):
            partition_dims = [dim for dim in dims if dim in da.dims]
            _write_partition_dataarray(da, store, ranks, partition_dims, rank)


def _write_partition_dataset_via_collected_variables(
    ds: xr.Dataset, store: str, ranks: int, dims: Sequence[Hashable], rank: int
):
    collected_by_partition = _collect_by_partition(ds, ranks, dims, rank)
    for partition, d in collected_by_partition:
        if partition is not None:
            d.isel(partition).to_zarr(store, region=partition)


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
            blocks = np.prod(block_sizes)
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
        if rank >= ranks:
            raise ValueError(f"Rank {rank} is greater than maximum rank {ranks - 1}.")

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

    def write(
        self,
        store: str,
        ranks: int,
        dims: Sequence[Hashable],
        rank: int,
        collect_variable_writes: bool = False,
    ):
        self.to_dataset().partition.write(
            store, ranks, dims, rank, collect_variable_writes
        )

    def mappable_write(
        self,
        store: str,
        ranks: int,
        dims: Sequence[Hashable],
        collect_variable_writes: bool = False,
    ) -> Callable[[int], None]:
        return self._obj.to_dataset().partition.mappable_write(
            store, ranks, dims, collect_variable_writes
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

    def initialize_store(
        self,
        store: str,
        inner_chunks: Optional[Dict[Hashable, int]] = None,
        mode: Optional[str] = None,
        zarr_format: Optional[int] = None,
    ):
        """Initialize a zarr store for partitioned writes.

        The ``inner_chunks`` and ``zarr_format`` parameters provided here
        will automatically be applied in the ``write`` step, as they are
        encoded on disk in the initialization process.

        Parameters
        ----------
        store : str
            Path to zarr store.
        inner_chunks : dict (optional)
            Dictionary mapping dimension names to inner chunk sizes for writing
            a sharded zarr store. Outer chunks (a.k.a. shards) will be inferred
            from the dask chunks on the variables in the Dataset. If not
            provided, a standard unsharded zarr store will be written, whose
            chunks will correspond to the dask chunks.
        mode : str or None
            ``mode`` to pass through to :py:meth:`xarray.Dataset.to_zarr`.
        zarr_format : int or None
            ``zarr_format`` to pass through to :py:meth:`xarray.Dataset.to_zarr`.
        """
        ds = self._obj
        if inner_chunks is not None:
            if zarr_format == 2:
                raise ValueError(
                    "It is not possible to specify inner_chunks when zarr_format=2. "
                    "Sharded stores are only possible with zarr version 3."
                )
            ds = set_shards_and_chunks_encoding(ds, inner_chunks)
        ds.to_zarr(store, compute=False, mode=mode, zarr_format=zarr_format)

    def write(
        self,
        store: str,
        ranks: int,
        dims: Sequence[Hashable],
        rank: int,
        collect_variable_writes: bool = False,
    ):
        """Write a Dataset partition to disk on a given rank.

        Parameters
        ----------
        store : str
            Path to zarr store.
        ranks : int
            Total number of ranks available to partition across.
        dims : Sequence[Hashable]
            Dimensions to partition among; if a dimension is left out
            no partitions will be made along that dimension.
        rank : int
            Rank of process to write partition from.
        collect_variable_writes : bool
            Whether to collect data variables with like partition indexers
            together when writing data out to disk (default False).  It can
            be beneficial to set this to True if data variables in the Dataset
            have like chunk structure, and also share intermediate data.  An
            example of this would be two fields that derive from the same
            input data.  By default this input data would need be computed or
            loaded twice; with this option set to True, it the input data would
            only need to be computed or loaded once.  A caveat, however, is that
            it can increase memory usage.
        """
        if collect_variable_writes:
            f = _write_partition_dataset_via_collected_variables
        else:
            f = _write_partition_dataset_via_individual_variables
        f(self._obj, store, ranks, dims, rank)

    def mappable_write(
        self,
        store: str,
        ranks: int,
        dims: Sequence[Hashable],
        collect_variable_writes: bool = False,
    ) -> Callable[[int], None]:
        """Return a function that can write data for a partition on a rank.

        Parameters
        ----------
        store : str
            Path to zarr store.
        ranks : int
            Total number of ranks available to partition across.
        dims : Sequence[Hashable]
            Dimensions to partition among; if a dimension is left out
            no partitions will be made along that dimension.
        collect_variable_writes : bool
            Whether to collect data variables with like partition indexers
            together when writing data out to disk (default False).  It can
            be beneficial to set this to True if data variables in the Dataset
            have like chunk structure, and also share intermediate data.  An
            example of this would be two fields that derive from the same
            input data.  By default this input data would need be computed or
            loaded twice; with this option set to True, it the input data would
            only need to be computed or loaded once.  A caveat, however, is that
            it can increase memory usage.

        Returns
        -------
        function
        """
        if collect_variable_writes:
            f = _write_partition_dataset_via_collected_variables
        else:
            f = _write_partition_dataset_via_individual_variables
        return functools.partial(f, self._obj, store, ranks, dims)


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


def get_unchunked_variable_names(ds):
    unchunked = []
    for name, variable in ds.variables.items():
        if isinstance(variable.data, np.ndarray):
            unchunked.append(name)
    return unchunked


def get_unchunked_non_dimension_coord_names(ds):
    names = []
    for name, da in ds.coords.items():
        if name not in ds.dims and isinstance(da.data, np.ndarray):
            names.append(name)
    return names


def get_unchunked_data_var_names(ds):
    names = []
    for name, da in ds.data_vars.items():
        if isinstance(da.data, np.ndarray):
            names.append(name)
    return names


def validate_PartitionMapper_dataset(ds):
    unchunked_non_dimension_coords = get_unchunked_non_dimension_coord_names(ds)
    unchunked_data_vars = get_unchunked_data_var_names(ds)
    invalid_unchunked_vars = unchunked_non_dimension_coords + unchunked_data_vars
    if invalid_unchunked_vars:
        raise ValueError(
            f"The PartitionMapper approach does not support writing datasets that "
            f"contain unchunked non-dimension coordinates or data variables.  "
            f"Consider dropping or chunking these before initiating the write or "
            f"switching to the traditional xpartition writing approach.  The "
            f"variables in question are {invalid_unchunked_vars!r}."
        )


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
        validate_PartitionMapper_dataset(iOut)

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
        unchunked_variables = get_unchunked_variable_names(iOut)
        iOut.drop_vars(unchunked_variables).to_zarr(self.path, region=region)
        logging.info(f"Done writing {rank + 1}.")

    def __iter__(self):
        self._initialize_store()
        return iter(range(len(self.plan.input_partition)))


def get_inner_chunk_size(
    inner_chunks: Dict[Hashable, int], dim_sizes: Dict[Hashable, int], dim: Hashable
) -> int:
    chunk_size = inner_chunks.get(dim, dim_sizes[dim])

    if chunk_size > 0 or chunk_size == -1:
        chunk_size = chunk_size if chunk_size > 0 else dim_sizes[dim]
    else:
        raise ValueError(
            f"Inner chunk size must be greater than 0 or be equal to -1; got chunk "
            f"size {chunk_size} along dim {dim!r}."
        )
    return chunk_size


def get_inner_chunks_encoding(
    da: xr.DataArray, inner_chunks: Dict[Hashable, int]
) -> Tuple[int, ...]:
    shards = dict(zip(da.dims, get_chunks_encoding(da)))

    chunks = []
    for dim in da.dims:
        chunk_size = get_inner_chunk_size(inner_chunks, da.sizes, dim)
        if shards[dim] % chunk_size == 0:
            chunks.append(chunk_size)
        else:
            raise ValueError(
                f"Inner chunk size ({chunk_size}) for dimension {dim!r} does not "
                f"evenly divide shard size ({shards[dim]}) for DataArray "
                f"{da.name!r}."
            )
    return tuple(chunks)


def set_shards_and_chunks_encoding(
    ds: xr.Dataset, inner_chunks: Dict[Hashable, int]
) -> xr.Dataset:
    # Make a shallow copy to avoid mutating the encoding of the input dataset.
    ds = ds.copy(deep=False)

    for da in {**ds.coords, **ds.data_vars}.values():
        if isinstance(da.data, dask.array.Array):
            shards = get_chunks_encoding(da)
            chunks = get_inner_chunks_encoding(da, inner_chunks)
            da.encoding["shards"] = shards
            da.encoding["chunks"] = chunks
    return ds
