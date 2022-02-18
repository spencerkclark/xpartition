import multiprocessing
import os
import string

import dask
import numpy as np
import pytest
import xarray as xr

import xpartition


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
    da = xr.DataArray(data, dims=dims, name=name)
    if chunks is not None:
        chunks = {dim: chunk for dim, chunk in zip(dims, chunks)}
        da = da.chunk(chunks)
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


@pytest.mark.filterwarnings("ignore:Specified Dask chunks")
@pytest.mark.parametrize("ranks", [1, 2, 3, 5, 10, 11])
def test_dataset_mappable_write(tmpdir, ds, ranks):
    store = os.path.join(tmpdir, "test.zarr")
    ds.partition.initialize_store(store)

    with multiprocessing.get_context("spawn").Pool(ranks) as pool:
        pool.map(ds.partition.mappable_write(store, ranks, ds.dims), range(ranks))

    result = xr.open_zarr(store)
    xr.testing.assert_identical(result, ds)


@pytest.mark.parametrize("has_coord", [True, False])
@pytest.mark.parametrize(
    "original_chunks", [{"x": 2}, {"x": 2, "y": 5}], ids=lambda x: f"{x}"
)
def test_PartitionMapper_integration(tmpdir, has_coord, original_chunks):
    def func(ds):
        return ds.rename(z="new_name").assign_attrs(dataset_attr="fun")

    ds = xr.Dataset({"z": (["x", "y"], np.ones((5, 10)), {"an": "attr"})}).chunk(
        original_chunks
    )
    if has_coord:
        ds = ds.assign_coords(x=range(5))

    store = str(tmpdir)
    mapper = ds.z.partition.map(store, ranks=3, dims=["x"], func=func, data=ds)
    for rank in mapper:
        mapper.write(rank)

    written = xr.open_zarr(store)
    xr.testing.assert_identical(func(ds), written)


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
    result = xpartition._zeros_like_dataarray(da, override_chunks)
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
    ("a", "b"),
    [
        (
            {"a": slice(None, None, 3), "b": slice(1, 10, 2)},
            {"b": slice(1, 10, 2), "a": slice(None, None, 3)},
        ),
        (None, None),
    ],
)
def test_HashableIndexers(a, b):
    assert a == b
    hashable_indexers_a = xpartition.HashableIndexers(a)
    hashable_indexers_b = xpartition.HashableIndexers(b)
    assert hash(hashable_indexers_a) == hash(hashable_indexers_b)


class CountingScheduler:
    """Inspired by xarray
    
    https://github.com/pydata/xarray/blob/33fbb648ac042f821a11870ffa544e5bcb6e178f/xarray/tests/__init__.py#L97-L113
    """

    def __init__(self):
        self.total_computes = 0

    def __call__(self, dsk, keys, **kwargs):
        self.total_computes += 1
        return dask.get(dsk, keys, **kwargs)


@pytest.mark.parametrize(
    "indexers", [{"a": slice(None, None, 3), "b": slice(1, 10, 2)}, None]
)
def test_HashableIndexers_from_immutable_representation(indexers):
    hashable_indexers = xpartition.HashableIndexers(indexers)
    immutable_representation = hashable_indexers.immutable_representation
    result = xpartition.HashableIndexers.from_immutable_representation(
        immutable_representation
    )
    assert result == hashable_indexers


def test_dataset_mappable_write_minimizes_compute_calls(tmpdir):
    # This tests to ensure that calls to compute are minimized when writing
    # partitioned Datasets.  Previously, a compute was called separately for
    # each variable in the Dataset.  For fields that have common intermediates --
    # e.g. loading a particular variable from somewhere -- this is inefficient,
    # because it means these intermediates must be computed multiple times.
    store = os.path.join(tmpdir, "test.zarr")

    foo = _construct_dataarray((2, 9), (2, 3), "foo")
    bar = (2 * foo).rename("bar")
    ds = xr.merge([foo, bar])

    ds.partition.initialize_store(store)
    scheduler = CountingScheduler()

    with dask.config.set(scheduler=scheduler):
        ranks = 3
        for rank in range(ranks):
            ds.partition.write(store, ranks, ds.dims, rank)

        assert scheduler.total_computes == ranks

        result = xr.open_zarr(store)
        xr.testing.assert_identical(result, ds)
