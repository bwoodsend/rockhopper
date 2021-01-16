# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest
from cslug import ptr

from rockhopper import RaggedArray
from rockhopper._ragged_array import sub_enumerate

pytestmark = pytest.mark.order(4)


@pytest.mark.parametrize("n", [10, 1, 0, 200])
@pytest.mark.parametrize("id_max", [10, 1, 200])
def test_from_ids(n, id_max):
    ids = np.random.randint(0, id_max, n)

    # -- Test ``sub_enumerate()`` --
    counts, sub_ids = sub_enumerate(ids, id_max)
    starts = np.empty_like(counts)
    starts[0] = 0
    counts[:-1].cumsum(out=starts[1:])
    unique = starts[ids] + sub_ids
    # Doing the above should provide a unique location for each item.
    assert np.all(np.sort(unique) == np.arange(len(ids)))

    # -- Test ``RaggedArray.group_by()`` and ``RaggedArray.multi_from_ids()``--
    # Cheat a bit by generating data to be grouped based on its group number.
    # This way, the ragged array can be validated simply by testing:
    #    ragged[i] == f(i)
    # where f() is the made up function used to generate the data from ``ids``.

    # Create a basic ragged array with ``sqrt(ids)`` as its data.
    self = RaggedArray.group_by(np.sqrt(ids), ids, id_max)
    assert len(self) == id_max

    # Create 3 ragged arrays simultaneously with data ``ids``, ``ids * 2`` and
    # ``ids *3`` respectively.
    datas = ids, ids * 2, ids * 3
    times_1, times_2, times_3 = RaggedArray.groups_by(ids, *datas,
                                                      id_max=id_max)
    assert len(times_1) == len(times_2) == len(times_3) == id_max

    # Create a single 3D array using the same information as above.
    _3D = RaggedArray.group_by(np.array(datas).T, ids, id_max)
    assert len(_3D) == id_max
    assert _3D.itemshape == (3,)

    # Check the contents of each.
    for i in range(id_max):
        assert np.all(self[i] == np.sqrt(i))

        assert np.all(times_1[i] == i)
        assert np.all(times_2[i] == 2 * i)
        assert np.all(times_3[i] == 3 * i)

        assert np.all(_3D[i] == np.array([i, 2 * i, 3 * i]))


def test_group_by_input_normalisation_and_type_checking():
    id_max = 20
    # Generate random ``ids`` with at least one of each value.
    ids = np.append(np.random.randint(0, id_max, 30), np.arange(id_max))
    np.random.shuffle(ids)
    data = np.random.random(ids.shape)

    explicit = RaggedArray.group_by(data, ids, id_max, check_ids=False)
    implicit = RaggedArray.group_by(data, ids)

    assert len(implicit) == len(explicit) == id_max
    assert np.all(implicit.starts == explicit.starts)

    with pytest.raises(IndexError):
        RaggedArray.group_by(data, ids, id_max - 1)
    with pytest.raises(IndexError):
        RaggedArray.group_by(data, ids - 1)
    shifted = RaggedArray.group_by(data, ids + 1)
    assert len(shifted) == id_max + 1
    assert len(shifted[0]) == 0
    assert np.all(shifted[1:].starts == explicit.starts)
