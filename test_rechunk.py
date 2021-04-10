import xarray
import numpy as np

from rechunk import *

def test_end_to_end(tmpdir):
    arr = xarray.DataArray(np.arange(50*100).reshape((100, 50)), dims=["x", "y"])
    arr = arr.chunk({"x": 1, "y": 50})
    rechunk = arr.chunk({"x": 10, "y": 10})

    in_partition = arr.partition.partition(100, ["x", "y"])
    out_partition = rechunk.partition.partition(50, ["x", "y"])

    msgs = send_messages(arr, in_partition, out_partition)
    groups = groupby(msgs)
    combined = {j: combine(out_partition[j], vals) for j, vals in groups}

    path = str(tmpdir)
    rechunk.to_dataset(name="a").partition.initialize_store(path)
    for j, ds in combined.items():
        ds.to_dataset(name="a").to_zarr(path, region=out_partition[j])
    
    loaded = xarray.open_zarr(path)
    assert loaded.chunks["x"] == (10,)*10
    assert loaded.chunks["y"] == (10,)*5
    xarray.testing.assert_equal(rechunk.load(), loaded['a'].load())




