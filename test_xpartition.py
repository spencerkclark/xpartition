import multiprocessing
import os
import string

import pytest
import numpy as np
import xarray as xr

import xpartition


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


def test_block_indices(da):
    blocks = xpartition.block_indices(da)
    blocks_shape = tuple(blocks.sizes[dim] for dim in da.dims)
    stacked_blocks = blocks.stack(block=[dim for dim in da.dims])
    for i in range(stacked_blocks.sizes["block"]):
        indexers = xpartition.block_to_slices(stacked_blocks.isel(block=i))
        dask_blocks_indices = np.unravel_index(i, blocks_shape)
        block_data_via_xarray = da.isel(indexers).data.compute()
        block_data_via_dask = da.data.blocks[dask_blocks_indices].compute()
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
def test_merge_blocks(subset):
    shape = (4, 3, 2, 7)
    chunks = (1, 2, 1, 5)
    dims = list(string.ascii_lowercase[: len(shape)])
    chunks = {dim: chunk for dim, chunk in zip(dims, chunks)}
    data = np.random.random(shape)
    da = xr.DataArray(data, dims=dims, name="foo").chunk(chunks)

    blocks = xpartition.block_indices(da)
    blocks_subset = blocks.isel(subset)
    merged_block = xpartition.merge_blocks(blocks_subset, da.dims)
    merged_block = xpartition.block_to_slices(merged_block)
    xarray_subset = da.isel(merged_block).data.compute()
    dask_subset = da.data.blocks[tuple(s for s in subset.values())].compute()

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
