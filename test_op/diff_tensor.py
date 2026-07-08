import os
import sys
import struct

import numpy as np

def compute_diff(array1, ref_array2, idx_cnt=-1):
    assert (
        array1.shape == ref_array2.shape
    ), f"Shape mismatch between arrays, {array1.shape} and {ref_array2.shape}"

    abs_diff = np.abs(array1 - ref_array2)

    max_abs_diff = abs_diff.max()
    range_ref = np.abs(ref_array2).max()

    max_rel_diff = max_abs_diff / (range_ref.max() + 1e-6)

    idx_cnt = min(idx_cnt, abs_diff.size)
    if idx_cnt <= 0:
        idx_cnt = abs_diff.size

    flat_indices = np.argsort(abs_diff.ravel())[-idx_cnt:][::-1]

    top_indices = np.array(np.unravel_index(flat_indices, abs_diff.shape)).T

    return max_abs_diff, max_rel_diff, top_indices


def print_diff(arr_act, arr_ref, tag_act, tag_ref, abs_diff, rel_diff, top_indices, eps):
    if rel_diff < eps:
        print(f"{tag_act} vs {tag_ref} Pass!")
        return

    print("-" * 65)
    print(f"{tag_act} vs {tag_ref}, abs err {abs_diff}, ref err {rel_diff}")

    print(f"{'Index':<20} {tag_act:<15} {tag_ref:<15} {'Abs Diff':<15}")

    for idx in top_indices:
        act_val = arr_act[tuple(idx)]
        ref_val = arr_ref[tuple(idx)]
        idx_diff = abs(act_val - ref_val)

        idx_str = str(tuple(idx))

        if isinstance(act_val, (int, np.integer)):
            print(f"{idx_str:<20} {act_val:<15} {ref_val:<15} {idx_diff:<15}")
        else:
            print(f"{idx_str:<20} {act_val:<15.6f} {ref_val:<15.6f} {idx_diff:<15.6f}")

