import multiprocessing
import os
import string

import dask
import numpy as np
import pytest
import xarray as xr
import zarr

import xpartition

from xpartition.xarray_utils import get_chunks_encoding, CountingScheduler
from xpartition.xpartition import (
    _zeros_like_dataarray,
    freeze_indexers,
    get_inner_chunk_size,
    get_inner_chunks_encoding,
    get_unchunked_data_var_names,
    get_unchunked_non_dimension_coord_names,
    get_unchunked_variable_names,
    unfreeze_indexers,
    validate_PartitionMapper_dataset,
)


@pytest.mark.parametrize(
    ("block_indexers", "expected", "exception"),
    [
        ({"x": slice(0, 3)}, {"x": slice(0, 6)}, None),
        ({"x": slice(1, 2)}, {"x": slice(2, 5)}, None),
        ({"x": slice(-3, -2)}, {"x": slice(0, 2)}, None),
        ({"x": slice(-3, -1)}, {"x": slice(0, 5)}, None),
        ({"x": slice(-3, None)}, {"x": slice(0, 6)}, None),
        ({"x": slice(None, 1)}, {"x": slice(0, 2)}, None),
        ({"x": slice(0, 10)}, {"x": slice(0, 6)}, None),
        ({"x": slice(-10, None)}, {"x": slice(0, 6)}, None),
        ({"x": slice(None, None)}, {"x": slice(0, 6)}, None),
        ({"x": slice(10, 12)}, {"x": slice(6, 6)}, None),
        ({"x": slice(2, 1)}, {"x": slice(5, 2)}, None),
        ({"x": 1}, {"x": slice(2, 5)}, None),
        ({"x": -1}, {"x": slice(5, 6)}, None),
        ({"x": -2}, {"x": slice(2, 5)}, None),
        ({"x": np.int32(2)}, {"x": slice(5, 6)}, None),
        ({"x": slice(0, 3), "y": 1}, {"x": slice(0, 6), "y": slice(3, 4)}, None),
        ({"x": 4}, None, IndexError),
        ({"x": -4}, None, IndexError),
        ({"z": 1}, None, KeyError),
        ({"x": slice(None, None, 2)}, None, NotImplementedError),
        ({"x": 2.0}, None, ValueError),
    ],
    ids=lambda x: f"{x}",
)
def test_indexers(block_indexers, expected, exception):
    data = dask.array.zeros((6, 4), chunks=((2, 3, 1), (3, 1)))
    da = xr.DataArray(data, dims=["x", "y"])
    if exception is None:
        result = da.blocks.indexers(**block_indexers)
        assert result == expected
    else:
        with pytest.raises(exception):
            da.blocks.indexers(**block_indexers)


def test_isel():
    data = dask.array.random.random((6, 4), chunks=((2, 3, 1), (3, 1)))
    da = xr.DataArray(data, dims=["x", "y"])

    result = da.blocks.isel(x=slice(1, 2), y=1).data.compute()
    expected = data.blocks[1:2, 1].compute()

    np.testing.assert_array_equal(result, expected)


@pytest.mark.filterwarnings("ignore:Specified Dask chunks")
@pytest.mark.parametrize("ranks", [1, 2, 3, 5, 10, 11])
def test_dataarray_mappable_write(tmpdir, da, ranks):
    store = os.path.join(tmpdir, "test.zarr")
    ds = da.to_dataset()
    ds.to_zarr(store, compute=False)

    with multiprocessing.get_context("spawn").Pool(ranks) as pool:
        pool.map(da.partition.mappable_write(store, ranks, da.dims), range(ranks))

    result = xr.open_zarr(store)
    xr.testing.assert_identical(result, ds)


SHAPE_AND_CHUNK_PAIRS = [
    ((5,), (1,)),
    ((5,), (2,)),
    ((5,), (5,)),
    ((2, 5), (1, 1)),
    ((2, 5), (2, 1)),
    ((2, 5), (2, 2)),
    ((2, 5), (2, 4)),
    ((2, 5), (2, 5)),
    ((2, 1, 6), (1, 1, 1)),
    ((2, 1, 6), (1, 1, 2)),
    ((2, 1, 6), (2, 1, 2)),
    ((2, 1, 6), (2, 1, 5)),
    ((2, 3, 4, 5), (1, 1, 1, 1)),
    ((2, 3, 4, 5), (2, 1, 3, 3)),
]


@pytest.fixture(params=SHAPE_AND_CHUNK_PAIRS, ids=lambda x: str(x))
def da(request):
    shape, chunks = request.param
    name = "foo"
    return _construct_dataarray(shape, chunks, name)


def _construct_dataarray(shape, chunks, name):
    dims = list(string.ascii_lowercase[: len(shape)])
    data = np.random.random(shape)
    coords = [range(length) for length in shape]
    da = xr.DataArray(data, dims=dims, name=name, coords=coords)
    if chunks is not None:
        chunks = {dim: chunk for dim, chunk in zip(dims, chunks)}
        da = da.chunk(chunks)

        # Add coverage for chunked coordinates
        chunked_coord_name = f"{da.name}_chunked_coord"
        da = da.assign_coords({chunked_coord_name: da.chunk(chunks)})
    return da


ALIGNED_SHAPE_AND_CHUNK_PAIRS = [
    ((5,), (1,)),
    ((5,), (2,)),
    ((5,), (5,)),
    ((5, 2), (1, 1)),
    ((5, 2), (1, 2)),
    ((5, 2), (2, 2)),
    ((5, 2), (4, 2)),
    ((5, 2), (5, 2)),
    ((5, 2, 6), (1, 1, 1)),
    ((5, 2, 6), (1, 1, 2)),
    ((5, 2, 6), (2, 1, 2)),
    ((5, 2, 6), (2, 2, 5)),
]


@pytest.fixture
def ds():
    unchunked_dataarrays = []
    for i, (shape, chunks) in enumerate(ALIGNED_SHAPE_AND_CHUNK_PAIRS):
        da = _construct_dataarray(shape, None, f"unchunked_{i}")
        unchunked_dataarrays.append(da)

    chunked_dataarrays = []
    for i, (shape, chunks) in enumerate(ALIGNED_SHAPE_AND_CHUNK_PAIRS):
        da = _construct_dataarray(shape, chunks, f"chunked_{i}")
        chunked_dataarrays.append(da)

    return xr.merge(unchunked_dataarrays + chunked_dataarrays)


def get_files(directory):
    names = os.listdir(directory)
    files = []
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            files.append(path)
    return files


def checkpoint_modification_times(store, variables):
    times = {}
    for variable in variables:
        directory = os.path.join(store, variable)
        files = get_files(directory)
        for file in files:
            times[file] = os.path.getmtime(file)
    return times


@pytest.mark.filterwarnings("ignore:Specified Dask chunks")
@pytest.mark.parametrize("ranks", [1, 2, 3, 5, 10, 11])
@pytest.mark.parametrize("collect_variable_writes", [False, True])
def test_dataset_mappable_write(tmpdir, ds, ranks, collect_variable_writes):
    unchunked_variables = get_unchunked_variable_names(ds)

    store = os.path.join(tmpdir, "test.zarr")
    ds.partition.initialize_store(store)

    # Checkpoint modification times of all files associated with unchunked
    # variables.  These should remain unchanged after initialization.
    expected_times = checkpoint_modification_times(store, unchunked_variables)

    with multiprocessing.get_context("spawn").Pool(ranks) as pool:
        pool.map(
            ds.partition.mappable_write(
                store, ranks, ds.dims, collect_variable_writes=collect_variable_writes
            ),
            range(ranks),
        )

    result = xr.open_zarr(store)

    # Check that dataset roundtrips identically.
    xr.testing.assert_identical(result, ds)

    # Checkpoint modification times of all files associated with unchunked
    # variables after writing the chunked variables.  The modification times of
    # the unchunked variables should be the same as before writing the chunked
    # variables.
    resulting_times = checkpoint_modification_times(store, unchunked_variables)
    assert expected_times == resulting_times


@pytest.mark.parametrize("has_coord", [True, False])
@pytest.mark.parametrize("has_chunked_coord", [True, False])
@pytest.mark.parametrize(
    "original_chunks", [{"x": 2}, {"x": 2, "y": 5}], ids=lambda x: f"{x}"
)
def test_PartitionMapper_integration(
    tmpdir, has_coord, has_chunked_coord, original_chunks
):
    def func(ds):
        return ds.rename(z="new_name").assign_attrs(dataset_attr="fun")

    ds = xr.Dataset({"z": (["x", "y"], np.ones((5, 10)), {"an": "attr"})}).chunk(
        original_chunks
    )
    if has_coord:
        ds = ds.assign_coords(x=range(5))
    if has_chunked_coord:
        chunked_coord = xr.DataArray(range(5), dims=["x"]).chunk({"x": 5})
        ds = ds.assign_coords(b=chunked_coord)

    unchunked_variables = get_unchunked_variable_names(ds)

    store = str(tmpdir)
    mapper = ds.z.partition.map(store, ranks=3, dims=["x"], func=func, data=ds)
    for i, rank in enumerate(mapper):
        if i == 0:
            expected_times = checkpoint_modification_times(store, unchunked_variables)
        mapper.write(rank)

    resulting_times = checkpoint_modification_times(store, unchunked_variables)
    assert expected_times == resulting_times

    written = xr.open_zarr(store)
    xr.testing.assert_identical(func(ds), written)


def test_PartitionMapper_integration_error():
    func = lambda ds: ds
    a = xr.DataArray(np.ones((5, 10)), [range(5), range(10)], ["x", "y"], name="a")
    b = a.copy(deep=True).rename("b").chunk({"x": 1})
    ds = xr.merge([a, b])
    mapper = ds.b.partition.map("store", ranks=3, dims=["x"], func=func, data=ds)
    with pytest.raises(ValueError, match="The PartitionMapper approach"):
        for rank in mapper:
            mapper.write(rank)


def test_partition_partition():
    # Partitions have two qualities which we test using a DataArray that
    # has all unique values
    ds = xr.Dataset({"z": (["x", "y"], np.arange(50).reshape((5, 10)))}).chunk({"x": 2})
    arr = ds["z"]

    n = 3
    regions = arr.partition.partition(n, dims=["x"])
    assert n == len(regions)

    def to_set(arr):
        return set(arr.values.ravel().tolist())

    # These are the properties of a partition
    # 1. sets in a partition are disjoint
    intersection = set.intersection(*[to_set(arr.isel(region)) for region in regions])
    assert intersection == set()

    # assert that the values cover the set
    # 2. the sets cover the original set
    union = set.union(*[to_set(arr.isel(region)) for region in regions])
    assert union == to_set(arr)


@pytest.mark.parametrize(
    ("original_chunks", "override_chunks", "expected_chunks"),
    [
        ({"x": 5, "y": 2}, None, ((5, 5), (2, 2, 2))),
        ({"x": 5, "y": 2}, {"y": 3}, ((5, 5), (3, 3))),
        ({"x": 5, "y": 2}, {"y": 3, "z": 1}, ((5, 5), (3, 3))),
    ],
    ids=lambda x: f"{x}",
)
@pytest.mark.parametrize("dtype", [float, int])
def test__zeros_like_dataarray(
    original_chunks, override_chunks, expected_chunks, dtype
):
    da = xr.DataArray(np.zeros((10, 6), dtype=dtype), dims=["x", "y"]).chunk(
        original_chunks
    )
    result = _zeros_like_dataarray(da, override_chunks)
    result_chunks = result.chunks
    assert result_chunks == expected_chunks
    assert result.dtype == da.dtype


def test_zeros_like():
    shape = (2, 4)
    dims = ["x", "y"]
    attrs = {"foo": "bar"}

    data1 = dask.array.random.random(shape)
    data2 = dask.array.random.randint(0, size=shape)
    data3 = dask.array.random.random(shape, chunks=(1, 1))

    da1 = xr.DataArray(data1, dims=dims, name="a", attrs=attrs)
    da2 = xr.DataArray(data2, dims=dims, name="b", attrs=attrs)
    da3 = xr.DataArray(data3, dims=dims, name="c", attrs=attrs)
    ds = xr.merge([da1, da2, da3])

    zeros1_data = dask.array.zeros(shape)
    zeros2_data = dask.array.zeros(shape, dtype=int)
    zeros3_data = dask.array.zeros(shape, chunks=(1, 1))

    zeros1 = xr.DataArray(zeros1_data, dims=dims, name="a", attrs=attrs)
    zeros2 = xr.DataArray(zeros2_data, dims=dims, name="b", attrs=attrs)
    zeros3 = xr.DataArray(zeros3_data, dims=dims, name="c", attrs=attrs)
    expected = xr.merge([zeros1, zeros2, zeros3])

    result = xpartition.zeros_like(ds)
    xr.testing.assert_identical(result, expected)

    for var in result:
        # assert_identical does not check dtype or chunks
        assert result[var].dtype == expected[var].dtype
        assert result[var].chunks == expected[var].chunks


def test_partition_indexers_invalid_rank_error():
    data = dask.array.zeros((6, 4), chunks=((6, 4)))
    da = xr.DataArray(data, dims=["x", "y"])
    with pytest.raises(ValueError, match="greater than maximum rank"):
        da.partition.indexers(1, 1, ["x"])


@pytest.mark.parametrize(
    ("unfrozen_indexers", "frozen_indexers"),
    [
        (
            {"a": slice(None, None, 3), "b": slice(1, 10, 2)},
            (("a", (None, None, 3)), ("b", (1, 10, 2))),
        ),
        (None, None),
    ],
    ids=lambda x: f"{x}",
)
def test_freeze_unfreeze_indexers(unfrozen_indexers, frozen_indexers):
    assert freeze_indexers(unfrozen_indexers) == frozen_indexers
    assert unfreeze_indexers(frozen_indexers) == unfrozen_indexers


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            {"a": slice(None, None, 3), "b": slice(1, 10, 2)},
            {"b": slice(1, 10, 2), "a": slice(None, None, 3)},
        ),
        (None, None),
    ],
    ids=lambda x: f"{x}",
)
def test_hashability_of_frozen_indexers(a, b):
    assert a == b
    frozen_indexers_a = freeze_indexers(a)
    frozen_indexers_b = freeze_indexers(b)

    # Despite having different key orders, the hashes of the frozen indexers
    # should be equal.
    assert hash(frozen_indexers_a) == hash(frozen_indexers_b)


@pytest.mark.parametrize(
    ("collect_variable_writes", "expected_computes"), [(False, 9), (True, 3)]
)
def test_dataset_mappable_write_minimizes_compute_calls(
    tmpdir, collect_variable_writes, expected_computes
):
    # This tests to ensure that calls to compute are minimized when writing
    # partitioned Datasets.  Previously, a compute was called separately for
    # each variable in the Dataset.  For fields that have common intermediates --
    # e.g. loading a particular variable from somewhere -- this is inefficient,
    # because it means these intermediates must be computed multiple times.  If
    # the option to collect_variable_writes is turned, however, we expect more
    # computes to be called (one for each partition and data variable in the
    # Dataset).
    store = os.path.join(tmpdir, "test.zarr")

    foo = _construct_dataarray((2, 9), (2, 3), "foo")
    bar = (2 * foo).rename("bar")
    ds = xr.merge([foo, bar])

    ds.partition.initialize_store(store)
    scheduler = CountingScheduler()

    with dask.config.set(scheduler=scheduler):
        ranks = 3
        for rank in range(ranks):
            ds.partition.write(store, ranks, ds.dims, rank, collect_variable_writes)

        assert scheduler.total_computes == expected_computes

    result = xr.open_zarr(store)
    xr.testing.assert_identical(result, ds)


@pytest.fixture()
def mixed_ds():
    dims = ["x_unchunked", "y_unchunked"]
    coords = [range(3), range(5)]
    data = np.zeros((3, 5))
    template_unchunked = xr.DataArray(data, coords, dims)
    template_chunked = xr.DataArray(data, coords, dims).chunk({"x_unchunked": 1})

    data_var_unchunked = template_unchunked.copy(deep=True).rename("data_var_unchunked")
    data_var_chunked = template_chunked.copy(deep=True).rename("data_var_chunked")
    coord_unchunked = template_unchunked.copy(deep=True).rename("coord_unchunked")
    coord_chunked = template_chunked.copy(deep=True).rename("coord_chunked")

    ds = xr.merge([data_var_chunked, data_var_unchunked])
    ds = ds.assign_coords(coord_unchunked=coord_unchunked, coord_chunked=coord_chunked)
    return ds


def test_get_unchunked_variable_names(mixed_ds):
    expected = {"x_unchunked", "y_unchunked", "data_var_unchunked", "coord_unchunked"}
    result = set(get_unchunked_variable_names(mixed_ds))
    assert result == expected


def test_get_unchunked_non_dimension_coord_names(mixed_ds):
    expected = {"coord_unchunked"}
    result = set(get_unchunked_non_dimension_coord_names(mixed_ds))
    assert result == expected


def test_get_unchunked_data_var_names(mixed_ds):
    expected = {"data_var_unchunked"}
    result = set(get_unchunked_data_var_names(mixed_ds))
    assert result == expected


def test_validate_PartitionMapper_dataset(mixed_ds):
    with pytest.raises(ValueError, match="The PartitionMapper approach"):
        validate_PartitionMapper_dataset(mixed_ds)


@pytest.mark.parametrize(
    ("mode", "raises_on_existing"), [(None, True), ("w-", True), ("w", False)]
)
def test_mode(tmpdir, ds, mode, raises_on_existing):
    store = os.path.join(tmpdir, "test.zarr")
    ds.to_zarr(store)

    if raises_on_existing:
        with pytest.raises(FileExistsError):
            ds.partition.initialize_store(store, mode=mode)
    else:
        ranks = 3
        ds.partition.initialize_store(store, mode=mode)
        for rank in range(ranks):
            ds.partition.write(store, ranks, ds.dims, rank)

        result = xr.open_zarr(store)
        xr.testing.assert_identical(result, ds)


@pytest.mark.parametrize("zarr_format", [None, 2, 3])
def test_zarr_format(tmpdir, ds, zarr_format):
    store = os.path.join(tmpdir, "test.zarr")

    ranks = 3
    ds.partition.initialize_store(store, zarr_format=zarr_format)
    for rank in range(ranks):
        ds.partition.write(store, ranks, ds.dims, rank)

    result = xr.open_zarr(store)
    xr.testing.assert_identical(result, ds)

    expected_zarr_format = 3 if zarr_format is None else zarr_format
    group = zarr.open_group(store)
    result_zarr_format = group.metadata.zarr_format
    assert result_zarr_format == expected_zarr_format


@pytest.mark.parametrize(
    ("chunks", "expected", "raises", "match"),
    [
        ({"a": 2, "b": 4}, (2, 4), False, None),
        ({"a": 2, "b": (1, 2, 2)}, None, True, "uniform chunk"),
        ({"a": 2, "b": (1, 1, 3)}, None, True, "Final chunk"),
    ],
    ids=lambda x: f"{x!r}",
)
def test_get_chunks_encoding(chunks, expected, raises, match):
    da = xr.DataArray(np.arange(10).reshape((2, 5)), dims=["a", "b"])
    da = da.chunk(chunks)
    if raises:
        with pytest.raises(ValueError, match=match):
            get_chunks_encoding(da)
    else:
        result = get_chunks_encoding(da)
        assert result == expected


@pytest.mark.parametrize(
    ("inner_chunks", "dim_sizes", "dim", "expected", "raises"),
    [
        ({"a": 1}, {"a": 2}, "a", 1, False),
        ({"a": -1}, {"a": 2}, "a", 2, False),
        ({"a": -2}, {"a": 2}, "a", None, True),
    ],
    ids=lambda x: f"{x!r}",
)
def test_get_inner_chunk_size(inner_chunks, dim_sizes, dim, expected, raises):
    if raises:
        with pytest.raises(ValueError, match="greater than 0"):
            get_inner_chunk_size(inner_chunks, dim_sizes, dim)
    else:
        result = get_inner_chunk_size(inner_chunks, dim_sizes, dim)
        assert result == expected


@pytest.mark.parametrize(
    ("inner_chunks", "raises"),
    [({"a": 1}, False), ({"a": 2}, True)],
    ids=lambda x: f"{x!r}",
)
def test_get_inner_chunks_encoding(inner_chunks, raises):
    da = xr.DataArray(np.arange(5), dims=["a"]).chunk({"a": 3})
    if raises:
        with pytest.raises(ValueError, match="evenly divide"):
            get_inner_chunks_encoding(da, inner_chunks)
    else:
        expected = (1,)
        result = get_inner_chunks_encoding(da, inner_chunks)
        assert result == expected


def test_sharded_store(tmpdir, ds):
    inner_chunks = {"a": 1, "b": 1, "c": 1}
    store = os.path.join(tmpdir, "sharded.zarr")

    ranks = 3
    ds.partition.initialize_store(store, inner_chunks=inner_chunks)
    for rank in range(ranks):
        ds.partition.write(store, ranks, ds.dims, rank)

    # Check that initialize_store and write do not mutate the encoding of
    # any of the variables in the original Dataset.
    for da in {**ds.coords, **ds.data_vars}.values():
        assert "shards" not in da.encoding
        assert "chunks" not in da.encoding

    # Check that the chunks and encoding of the loaded Dataset match our
    # expectations.
    result = xr.open_zarr(store)
    for name, original_da in {**ds.coords, **ds.data_vars}.items():
        result_da = result[name]

        if isinstance(result_da.data, dask.array.Array):
            stored_chunks = {}
            for dim, (size, *_) in zip(result_da.dims, result_da.chunks):
                stored_chunks[dim] = size

            if isinstance(original_da.data, dask.array.Array):
                expected_chunks = {dim: inner_chunks[dim] for dim in stored_chunks}
                expected_shards_encoding = get_chunks_encoding(ds[name])
                expected_chunks_encoding = get_chunks_encoding(result_da)
            else:
                expected_chunks = result_da.sizes
                expected_shards_encoding = None
                expected_chunks_encoding = get_chunks_encoding(result_da)

            assert stored_chunks == expected_chunks
            assert result_da.encoding["shards"] == expected_shards_encoding
            assert result_da.encoding["chunks"] == expected_chunks_encoding

    # Finally check that the written Dataset, modulo chunks, is identical
    # to the provided Dataset.
    xr.testing.assert_identical(result, ds)


def test_inner_chunks_zarr_format_2_error(tmpdir, ds):
    inner_chunks = {"a": 1, "b": 1, "c": 1}
    zarr_format = 2
    store = os.path.join(tmpdir, "sharded.zarr")

    with pytest.raises(ValueError, match="zarr_format=2"):
        ds.partition.initialize_store(
            store, inner_chunks=inner_chunks, zarr_format=zarr_format
        )
