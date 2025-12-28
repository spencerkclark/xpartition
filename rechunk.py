from typing import Mapping, Hashable, Optional
import xpartition
import xarray
import dask.array
from collections import defaultdict
import numpy as np

Region = Mapping[Hashable, slice]


def send_messages(arr, in_partition, out_partition):
    for in_region in in_partition:
        input = arr.isel(in_region).load()
        for j, out_region in enumerate(out_partition):
            message = _intersection(in_region, out_region)
            if message is not None:
                yield j, (
                    message,
                    input.isel(_local_region(in_region, message)),
                )


def groupby(msgs):
    store = defaultdict(list)
    for j, msg in msgs:
        store[j].append(msg)
    return store.items()


def combine(region: Region, arrs):
    sizes = _size(region)
    _, arr = arrs[0]
    dims = arr.dims
    shape = [sizes[dim] for dim in dims]

    output = np.zeros_like(arr, shape=shape)
    for global_region, arr in arrs:
        arr = arr.transpose(*dims)
        local_region = _local_region(region, global_region)
        arr_slices = tuple(local_region[dim] for dim in dims)
        output[arr_slices] = np.asarray(arr)

    return xarray.DataArray(output, dims=dims, attrs=arr.attrs)


def _local_slice(sl: slice, sl_small):
    return slice(sl_small.start - sl.start, sl_small.stop - sl.start)


def _local_region(region: Region, sub_region: Region):
    return {dim: _local_slice(region[dim], sub_region[dim]) for dim in region}


def _size(region: Region):
    return {key: s.stop - s.start for key, s in region.items()}


def _intersection(region1: Region, region2: Region) -> Optional[slice]:
    region = {key: _intersect_slice(region1[key], region2[key]) for key in region1}
    if any(i is None for i in region.values()):
        return None
    else:
        return region


def _intersect_slice(sl1, sl2) -> Optional[slice]:
    start = max(sl1.start, sl2.start)
    stop = min(sl1.stop, sl2.stop)
    if start < stop:
        return slice(start, stop)
    else:
        return None