import multiprocessing
import os
import string

import pytest
import numpy as np
import xarray as xr

import xpartition


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, True), (np.int32(1), True), (2.0, False), (np.float32(2), False)],
)
def test__is_integer(value, expected):
    result = xpartition._is_integer(value)
    assert result == expected


@pytest.mark.parametrize(
    ("indexers", "exception"),
    [
        ({"x": 0}, None),
        ({"x": 4}, None),
        ({"x": -4}, None),
        ({"x": 5}, IndexError),
        ({"x": -5}, IndexError),
        ({"x": 2.0}, ValueError),
        ({"y": 0}, KeyError),
    ],
    ids=lambda x: f"{x}",
)
def test__validate_indexers(indexers, exception):
    sizes = {"x": 5}
    if exception is not None:
        with pytest.raises(exception):
            xpartition._validate_indexers(indexers, sizes)
    else:
        xpartition._validate_indexers(indexers, sizes)


@pytest.mark.parametrize(
    ("indexers", "expected"),
    [
        ({"x": 5}, {"x": slice(5, 6)}),
        ({"x": -5}, {"x": slice(-5, -4)}),
        ({"x": slice(0, 2), "y": 5}, {"x": slice(0, 2), "y": slice(5, 6)}),
    ],
    ids=lambda x: f"{x}",
)
def test__convert_scalars_to_slices(indexers, expected):
    result = xpartition._convert_scalars_to_slices(indexers)
    assert result == expected


@pytest.mark.parametrize(
    ("block_indexers", "expected"),
    [
        ({"x": slice(0, 3)}, {"x": slice(0, 6)}),
        ({"x": slice(1, 2)}, {"x": slice(2, 5)}),
        ({"x": slice(-3, -2)}, {"x": slice(0, 2)}),
        ({"x": slice(-3, -1)}, {"x": slice(0, 5)}),
        ({"x": slice(-3, None)}, {"x": slice(0, 6)}),
        ({"x": slice(None, 1)}, {"x": slice(0, 2)}),
        ({"x": slice(0, 10)}, {"x": slice(0, 6)}),
        ({"x": slice(-10, None)}, {"x": slice(0, 6)}),
        ({"x": slice(None, None)}, {"x": slice(0, 6)}),
        ({"x": slice(10, 12)}, {"x": slice(6, 6)}),
    ],
    ids=lambda x: f"{x}",
)
def test__convert_block_indexers_to_array_indexers(block_indexers, expected):
    chunks = {"x": (2, 3, 1)}
    result = xpartition._convert_block_indexers_to_array_indexers(
        block_indexers, chunks
    )
    assert result == expected


def _construct_dataarray(shape, chunks, name):
    dims = list(string.ascii_lowercase[: len(shape)])
    data = np.random.random(shape)
    da = xr.DataArray(data, dims=dims, name=name)
    if chunks is not None:
        chunks = {dim: chunk for dim, chunk in zip(dims, chunks)}
        da = da.chunk(chunks)
    return da


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


def test_indexers_with_scalars(da):
    n_blocks = np.product([size for size in da.blocks.sizes.values()])
    for i in range(n_blocks):
        block_indices = np.unravel_index(i, da.blocks.shape)
        block_indexers = {dim: index for dim, index in zip(da.dims, block_indices)}
        array_indexers = da.blocks.indexers(**block_indexers)
        block_data_via_xarray = da.isel(array_indexers).data.compute()
        block_data_via_dask = da.data.blocks[block_indices].compute()
        np.testing.assert_array_equal(block_data_via_xarray, block_data_via_dask)

        # Test obtaining the array through blocks.isel
        block_data_via_xarray = da.blocks.isel(**block_indexers).data.compute()
        np.testing.assert_array_equal(block_data_via_xarray, block_data_via_dask)


@pytest.mark.parametrize(
    "subset",
    [
        {"a": slice(0, 1), "b": slice(0, 1), "c": slice(0, 1), "d": slice(0, 1)},
        {"a": slice(0, 2), "b": slice(0, 1), "c": slice(0, 1), "d": slice(0, 1)},
        {"a": slice(0, 1), "b": slice(0, 2), "c": slice(0, 1), "d": slice(0, 1)},
        {"a": slice(0, 1), "b": slice(0, 1), "c": slice(0, 2), "d": slice(0, 1)},
        {"a": slice(0, 1), "b": slice(0, 1), "c": slice(0, 1), "d": slice(0, 2)},
        {"a": slice(1, 2), "b": slice(1, 2), "c": slice(1, 2), "d": slice(1, 2)},
        {"a": slice(0, 3), "b": slice(0, 1), "c": slice(1, 2), "d": slice(1, 2)},
        {"a": slice(0, 3), "b": slice(0, 2), "c": slice(0, 1), "d": slice(0, 2)},
        {"a": slice(0, 4), "b": slice(0, 2), "c": slice(0, 2), "d": slice(0, 2)},
    ],
    ids=lambda x: str(x),
)
def test_indexers_with_slices(subset):
    shape = (4, 3, 2, 7)
    chunks = (1, 2, 1, 5)
    dims = list(string.ascii_lowercase[: len(shape)])
    chunks = {dim: chunk for dim, chunk in zip(dims, chunks)}
    data = np.random.random(shape)
    da = xr.DataArray(data, dims=dims, name="foo").chunk(chunks)

    indexers = da.blocks.indexers(**subset)
    xarray_subset = da.isel(indexers).data.compute()
    dask_subset = da.data.blocks[tuple(s for s in subset.values())].compute()
    np.testing.assert_array_equal(xarray_subset, dask_subset)

    # Test obtaining the array through blocks.isel
    xarray_subset = da.blocks.isel(**subset).data.compute()
    np.testing.assert_array_equal(xarray_subset, dask_subset)


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
def test_PartitionMapper_integration(tmpdir, has_coord):
    def func(ds):
        return ds.rename(z="new_name").assign_attrs(dataset_attr="fun")

    ds = xr.Dataset({"z": (["x", "y"], np.ones((5, 10)), {"an": "attr"})}).chunk(
        {"x": 2}
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
