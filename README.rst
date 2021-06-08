==========
rockhopper
==========

.. image::
    https://img.shields.io/badge/
    Python-%203.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%20PyInstaller-blue.svg

A *Ragged Array* class: 2D NumPy arrays containing rows of mismatching lengths.

* Free software: MIT license
* Source code: https://github.com/bwoodsend/rockhopper/
* Releases: https://pypi.org/project/rockhopper/
* Documentation: You are looking at it... 🤨

NumPy arrays are very powerful but its multidimensional arrays must be
rectangular (or cuboidal, hyper-cuboidal, tesseractal, ...).
A ``rockhopper.RaggedArray()`` wraps a 1D NumPy array into something resembling
a 2D NumPy array but with the *rectangular* constraint loosened.
i.e. The following is perfectly valid:

.. code-block:: python

    from rockhopper import ragged_array

    ragged = ragged_array([
        # Row with 4 items
        [1.2, 23.3, 4.1 , 12],
        # Row with 3 items
        [2.0, 3., 43.9],
        # Row with no items
        [],
        # Another row with 4 items
        [0.12, 7.2, 1.3, 42.9],
    ])

Under the hood,
rockhopper operations use NumPy vectorisation where possible
and C when not
so that performance is almost as good as normal NumPy
and still orders of magnitudes faster than pure Python list of lists
implementations.


Features
--------

It's early days for **rockhopper**.
Features have so far been added on an *as needed* basis
and consequently, its features list has some holes in it.
The following shows what **rockhopper** has, labelled with a ✓,
and what it doesn't (yet) have, labelled with a ✗.

* `Initialisation`_ from:
    - ✓ A ragged list of lists.
    - ✓ A flat contents array and a list of row lengths.
    - ✓ A flat contents array and a list of row start/ends.
* `Indexing and Slicing`_ (getting/setting support marked separately with a ``'/'`` divider):
    - `1D indices`_ ``ragged[rows]`` where:
        * ✓/✓: **rows** is an integer.
        * ✓/✗: **rows** is a list of integers, bool mask or slice.
    - `2D indices`_ ``ragged[rows, columns]`` where:
        * ✓/✓ **rows** is anything and **columns** is an integer or list of
          integers.
        * ✓/✗: **rows** is anything and **columns** is a bool mask or slice.
    - `3D (or higher) indices`_ ``ragged[x, y, z]`` (only applicable to `higher dimensional arrays`_) where:
        * ✓/✓ **x** is anything, **y** is an integer or list of integers, and
          **z** is anything.
        * ✗/✗: **x** is anything, and **y** is a bool mask or slice, and **z**
          is anything.
* Concatenation (joining multiple arrays together):
    - ✗ rows
    - ✗ columns
* Vectorisation - these will take a bit of head scratching to get working:
    - ✗ Applying arithmetic operations (e.g. ``ragged_array * 3``) so that the
      for loop is efficiently handled in NumPy.
    - ✗ Reverse ``__getitem__()``. i.e. ``regular_array[ragged_integer_array]``
      should create another ragged array whose contents are taken from
      ``regular_array``.
* `Export to standard types`_:
    - ✓ The ``tolist()`` method takes you back to a list of lists.
    - ✓ The ``to_rectangular_arrays()`` method converts to a list of regular
      rectangular arrays.
* `Serialisation and deserialisation`_:
    - ✓ Binary_ (``row-length|row-content`` format).
    - ✗ Ascii. (Saving this for a rainy day.)
    - ✓ Pickle_.
* ✓ Grouping_ data by some enumeration - similar to
  ``pandas.DataFrame.groupby()``.


Installation
------------

To install use the following steps:

1.  Think of a prime number between 4294967296 and 18446744073709551616,
2.  Multiply it by the diameter of your ear lobes,
3.  Negate it then take the square root,
4.  Subtract the number you first thought of,
5.  Run the following in some flavour of terminal::

        pip install rockhopper

Pre-built binary wheels (i.e. easy to install) are shipped for:

* Linux distributions based on glibc whose architecture NumPy also ships
  prebuilt wheels for (which can be seen `here
  <https://pypi.org/project/numpy/#files>`_)
* Windows 64 and 32 bit
* macOS >=10.6 on ``x86_86`` or ``arm64``

Other supported and tested platforms (which ``wheel`` lacks support for) are:

* musl based Linux (requires gcc_ to build)
* FreeBSD (requires clang_ or gcc_ to build)

On these platforms, **rockhopper** should build from and install out the box
if your first install the appropriate C compiler.

.. _many linux project: https://quay.io/organization/pypa
.. _gcc: https://gcc.gnu.org/
.. _clang: https://clang.llvm.org/


Usage
-----


Initialisation
..............

The easiest way to make a ragged array is from a nested list using
``rockhopper.ragged_array()``.

.. code-block:: python

    from rockhopper import ragged_array

    ragged = ragged_array([
        [1, 2, 3],
        [2, 43],
        [34, 32, 12],
        [2, 3],
    ])

In this form, what goes in is what comes out.

.. code-block:: python

    >>> ragged
    RaggedArray.from_nested([
        [1, 2, 3],
        [ 2, 43],
        [34, 32, 12],
        [2, 3],
    ])

As the repr implies, the output is of type ``rockhopper.RaggedArray`` and
the ``ragged_array()`` function is simply a shortcut for
``RaggedArray.from_nested()`` which you may call directly if you prefer.
Data types (the `numpy.dtype`_) are implicit but may be overrode using the
**dtype** parameter.


.. code-block:: python

    >>> ragged_array([
    ...     [1, 2, 3],
    ...     [2, 43],
    ...     [34, 32, 12],
    ...     [2, 3],
    ... ], dtype=float)
    RaggedArray.from_nested([
        [1., 2., 3.],
        [ 2., 43.],
        [34., 32., 12.],
        [2., 3.],
    ])


.. _`numpy.dtype`: https://numpy.org/doc/stable/reference/arrays.dtypes.html

Alternative ways to construct are from flat contents and row lengths:

.. code-block:: python

    from rockhopper import RaggedArray

    # Creates exactly the same array as above.
    ragged = RaggedArray.from_lengths(
        [1, 2, 3, 2, 43 34, 32, 12, 2, 3],  # The array contents.
        [3, 2, 3, 2],  # The length of each row.
    )

Or at a lower level, a flat contents array and an array of row *bounds* (the
indices at which one row ends and next one begins).
As with regular Python ``range()`` and slices, a row includes the starting index
but excludes the end index.

.. code-block:: python

    # Creates exactly the same array as above.
    ragged = RaggedArray(
        [1, 2, 3, 2, 43 34, 32, 12, 2, 3],  # The array contents again.
        [0, 3, 5, 8, 10],  # The start and end of each row.
    )

Or at an even lower level, a flat contents array and separate arrays for where
each row starts and each row ends.
This form reflects how the ``RaggedArray`` class's internals are structured.

.. code-block:: python

    # And creates the same array as above again.
    ragged = RaggedArray(
        [1, 2, 3, 2, 43 34, 32, 12, 2, 3],  # The array contents.
        [0, 3, 5, 8],  # The starting index of each row.
        [3, 5, 8, 10],  # The ending index of each row.
    )

This last form is used internally for efficient slicing but isn't expected to be
particularly useful for day to day usage.
With this form, rows may be in mixed orders, have gaps between them or overlap.

.. code-block:: python

    # Creates a weird array.
    ragged = RaggedArray(
        range(10),  # The array contents.
        [6, 3, 4, 1, 2],  # The starting index of each row.
        [9, 5, 8, 2, 2],  # The ending index of each row.
    )

Externally, the fact that rows share data or have gaps in between is invisible.

.. code-block:: python

    >>> ragged
    RaggedArray.from_nested([
        [6, 7, 8],
        [3, 4],
        [4, 5, 6, 7],
        [1],
        [],
    ])


Higher Dimensional Arrays
*************************

Rockhopper is very much geared towards 2D ragged arrays, however,
one permutation of higher dimensional ragged arrays is allowed:
A ragged array's rows can be multidimensional rather than a 1D arrays.

Construction works more or less as you'd expect.
The following shows 3 different ways to create the same multidimensional ragged
array.

.. code-block:: python

    import numpy as np
    from rockhopper import ragged_array, RaggedArray

    # Construct from nested lists.
    from_nested = ragged_array([
        [[0,  1], [2, 3]],
        [[4, 5]],
        [[6, 7], [8, 9], [10, 11]],
        [[12, 13]],
    ])

    # Construction from flat contents and either ...
    flat = np.array([
        [0,  1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]
    ])
    # ... row lengths, ...
    from_lengths = RaggedArray.from_lengths(flat, [2, 1, 3, 2])
    # ... or row bounds.
    from_bounds = RaggedArray(flat, [0, 2, 3, 6, 7])


Structured Arrays
*****************

Ragged arrays may also use a `structured data type
<https://numpy.org/doc/stable/user/basics.rec.html>`_.
For this, explicitly setting the **dtype** parameter is mandatory when using
the ``ragged_array()`` constructor.
Otherwise NumPy will cast everything to one compatible type (usually ``str``).

.. code-block:: python

    ragged = ragged_array([
        [("abc", 3), ("efg", 5)],
        [("hij", 1)],
        [("klm", 13), ("nop", 99), ("qrs", 32)],
    ], dtype=[("foo", str, 3), ("bar", int)])

However, this feature is only half-formed because ``ragged["foo"]`` requires
internal support for strided flat arrays (which rockhopper currently lacks).


Indexing and Slicing
....................

Most forms of ``__getitem__()`` and ``__setitem__()``
(i.e. ``ragged[x]`` and ``ragged[x] = y``)
are supported and mirror the semantics of `NumPy indexing`_.

There are a few general rules of thumb for what isn't supported:

* When a get operation returns another ragged array, the corresponding set
  operation is not implemented. This would require implementing vectorisation to
  work.
* If a 2D index ``ragged[x, y]`` gives another ragged array, then neither
  getting or setting is supported for >2D indices which start with said 2D index
  ``ragged[x, y, z]``. This would require internal support for letting
  ``ragged.flat`` be strided.
* Ragged arrays can not be used as indices. ``arr[ragged]`` will fail
  irregardless or whether ``arr`` is ragged or not.
* Under no circumstances will writing to a ragged array be allowed to change
  its overall length or the length of one of its rows.

In all cases except where indicated otherwise,
indexing returns original data - not copies.
If you later write to either the ragged array itself or a slice taken from it,
then the other will change too.

.. _NumPy indexing: https://numpy.org/doc/stable/reference/arrays.indexing.html


1D indices
**********

Indexing will all be shown by examples.
Here is an unimaginative ragged array to play with.

.. code-block:: python

    from rockhopper import ragged_array

    ragged = ragged_array([
        [1, 2, 3, 4],
        [5, 6],
        [7, 8, 9],
        [10, 11, 12, 13],
    ])

1D indexing with individual integers gives single rows as regular arrays.

.. code-block:: python

    >>> ragged[2]
    array([7, 8, 9])
    >>> ragged[3]
    array([10, 11, 12, 13])

But indexing with a slice, integer array or bool mask gives another ragged
array.

.. code-block:: python

    >>> ragged[::2]
    RaggedArray.from_nested([
        [1, 2, 3, 4],
        [7, 8, 9],
    ])
    >>> ragged[[2, -1]]
    RaggedArray.from_nested([
        [7, 8, 9],
        [10, 11, 12, 13],
    ])


This is true even if all rows happen to be the same length.


2D indices
**********

2D indexing ``ragged[rows, columns]`` gives individual cells.
Arrays of indices, slices and bool masks may also be used instead of single
numbers.
Using the same boring ragged array `as above <#d-indices>`_:

.. code-block:: python

    # Individual indices.
    >>> ragged[0, 0], ragged[0, 1], ragged[0, 2]
    (1, 2, 3)

    # Arrays of indices.
    >>> ragged[0, [0, 1, -1]]
    array([1, 2, 4])
    >>> ragged[0, [[1, 2], [0, 2]]]
    array([[2, 3],
           [1, 3]])
    >>> ragged[[0, 3, 2], [2, 3, 1]]
    array([ 3, 13,  8])

    # Slices as row numbers (including the null slice [:]).
    >>> ragged[:, 0]
    array([ 1,  5,  7, 10])
    >>> ragged[2:, -1]
    array([ 9, 13])

    # Again, multiple column numbers may be given.
    # The following gets the first and last element from each row.
    >>> ragged[:, [0, -1]]
    array([[ 1,  4],
           [ 5,  6],
           [ 7,  9],
           [10, 13]])

    # If the second index is a slice or bool mask, the output is a ragged array.
    # Even if each row is of the same length.
    >>> ragged[:, :2]
    RaggedArray.from_nested([
        [1, 2],
        [5, 6],
        [7, 8],
        [10, 11],
    ])

If the second index is not a slice then the the output of getitem is a copy and
does not share memory with the parent ragged array.


3D (or higher) indices
**********************

`Higher Dimensional Arrays`_ can be sliced using 3 indices (or more).

Using another uninspiring enumeration example - this time a 3D array:

.. code-block:: python

    ragged = ragged_array([
        [[ 0,  1,  2], [ 3,  4,  5]],
        [[ 6,  7,  8], [ 9, 10, 11]],
        [[12, 13, 14], [15, 16, 17], [18, 19, 20]],
        [[21, 22, 23]],
    ])

3D arrays follow the same indexing rules as 2D arrays except that each **cell**
is actually another array.

.. code-block:: python

    >>> ragged[0, 1]
    array([3, 4, 5])

And a triplet of indices are used to access individual elements.

.. code-block:: python

    >>> ragged[2, 0, 1]
    13


Export to standard types
........................

No matter how many features I cram in to make ragged arrays more interchangeable
with normal ones,
you'll probably want to get back into regular array territory at the first
opportunity.
**rockhopper** comes with a few ways to do so.

First, let us create a ragged array to export:

.. code-block:: python

    from rockhopper import ragged_array
    ragged = ragged_array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8],
        [9, 10],
        [11, 12, 13],
    ])


To list of lists
****************

The ``tolist()`` method converts back to nested lists (like those used to build
the array in the first place).

.. code-block:: python

    >>> ragged.tolist()
    [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10], [11, 12, 13]]


To list of homogenous arrays
****************************

When a ragged array is either not very ragged (row lengths are mostly the same)
or not ragged at all (rows are all the same length),
it's often helpful to split it on rows of differing lengths,
giving a sequence of standard rectangular arrays which can be ``for loop``\ -ed
over.
Do this with the ``to_rectangular_arrays()`` method.

.. code-block:: python

    >>> ragged.to_rectangular_arrays()
    [array([[1, 2, 3],
            [4, 5, 6]]),
     array([[7, 8],
            [9, 10]]),
     array([[11, 12, 13]])]

In the somewhat unlikely event that you don't care about the order the rows
appear in,
set the **reorder** option to allow it to presort the rows into ascending
lengths so as to minimize fragmentation.

.. code-block:: python

    >>> sort_args, arrays = ragged.to_rectangular_arrays(reorder=True)
    # The numpy.argsort() arguments are returned in case you want them.
    >>> sort_args
    array([2, 3, 0, 1, 4])
    # By sorting, only 2 arrays are needed rather than 3.
    >>> arrays
    [array([[ 7,  8],
            [ 9, 10]]),
     array([[ 1,  2,  3],
            [ 4,  5,  6],
            [11, 12, 13]])]


Serialisation and Deserialisation
.................................

Ragged arrays may be converted to bytes and back again
which can be read from or written to files.


Binary
******

Currently **rockhopper** knows of exactly one binary format:
The highly typical, but hopelessly un-NumPy-friendly::

    row-length | row-content | row-length | row-content

binary form often found in 3D graphics
where ``row-length`` may be any unsigned integer type of either byteorder,
``row-content`` may be of any data type or byteorder,
and there are no delimiters or metadata anywhere.

For this format ``RaggedArray()`` provides a ``loads()`` method for reading
and a ``dumps()`` method for writing.

Some examples:

.. code-block:: python

    # Write using:
    #  - Row contents: The current data type (ragged.dtype) and endian.
    #  - Row lengths: ``numpy.intc`` native endian
    # Note that the output is a memoryview() which is generally interchangeable
    # with bytes(). This may still be written to a file with the usual
    # ``fh.write()``.
    dumped = ragged.dumps()

    # Read back using:
    #  - Row contents: The same dtype used to write it
    #  - Row lengths: ``numpy.intc`` native endian
    ragged, bytes_consumed = RaggedArray.loads(dumped, ragged.dtype)

    # Write then read using:
    #  - Row contents: Big endian 8-byte floats
    #  - Row lengths: Little endian 2-byte unsigned integers
    dumped = ragged.astype(">f8").dumps(ldtype="<u2")
    ragged, bytes_consumed = RaggedArray.loads(dumped, ">f8", ldtype="<u2")

By default, ``loads()`` will keep adding rows until it hits the end of the byte
array that it's parsing.
The ``bytes_consumed`` (a count of how many bytes from ``dumped`` where used)
will therefore always satisfy ``bytes_consumed == len(dumped)``.

Some file formats contain a serialised ragged array embedded inside a larger
file but don't specify how many bytes belong to
the ragged array and how many belong to whatever comes afterwards.
Instead they specify how many rows there should be.
To read such data use the **rows** keyword argument.

.. code-block:: python

    # Read a 20 row ragged array of floats from a long ``bytes()`` object called
    # **blob**. Will raise an error if it runs out of data.
    ragged, bytes_consumed = ragged.loads(blob, "f8", rows=20)

    # ``bytes_consumed`` indicates where the ragged array stopped.
    rest_of_blob = blob[bytes_consumed:]


Pickle
******

If you don't need other programs to be able to read the output then bog-standard
pickle works too.

.. code-block:: python

    >>> import pickle
    >>> arr = ragged_array([
    ...    ["cake", "biscuits"],
    ...    ["socks"],
    ...    ["orange", "lemon", "pineapple"],
    ... ])
    >>> pickle.loads(pickle.dumps(arr))
    RaggedArray.from_nested([
        ["cake", "biscuits"],
        ["socks"],
        ["orange", "lemon", "pineapple"],
    ])


Grouping
........

Arbitrary data may be grouped by some group enumeration into a ragged array so
that each data element appears on the row of its group number.

For example, to group the people in the following array...

.. code-block:: python

    people = np.array([
        ("Bob", 1),
        ("Bill", 2),
        ("Ben", 0),
        ("Biff", 1),
        ("Barnebas", 0),
        ("Bubulous", 1),
        ("Bofflodor", 2),
    ], dtype=[("name", str, 20), ("group number", int)])

... by the **group number** field use:

.. code-block:: python

    >>> from rockhopper import RaggedArray
    >>> RaggedArray.group_by(people, people["group number"])
    RaggedArray.from_nested([
        [('Ben', 0), ('Barnebas', 0)],
        [('Bob', 1), ('Biff', 1), ('Bubulous', 1)],
        [('Bill', 2), ('Bofflodor', 2)],
    ])

As you can hopefully see,

- all the names given a **group number** 0 appear in row 0,
- all the names given a **group number** 1 appear in row 1,
- and all the names given a **group number** 1 appear in row 2.

At this point you probably no longer care about the **group number** field,
in which case, group only the **name** field:

.. code-block:: python

    >>> RaggedArray.group_by(people["name"], people["group number"])
    RaggedArray.from_nested([
        ['Ben', 'Barnebas'],
        ['Bob', 'Biff', 'Bibulous'],
        ['Bill', 'Bofflodor'],
    ])


Enumerating classes
*******************

The above assumes that the parameter you wish to group by is just an
enumeration.
If this is not the case, and you're not already sick of software written by me,
then you may use a `hirola.HashTable()
<https://github.com/bwoodsend/Hirola#hirola>`_ to efficiently enumerate the
parameter to group by.

For example, to group this list of animals by their animal class:

.. code-block:: python

    animals = np.array([
        ("cow", "mammal"),
        ("moose", "mammal"),
        ("centipede", "insect"),
        ("robin", "bird"),
        ("spider", "insect"),
        ("whale", "mammal"),
        ("woodpecker", "bird"),
    ], dtype=[("name", str, 15), ("class", str, 15)])

Use something like:

.. code-block:: python

    >>> from hirola import HashTable
    >>> animal_classes = HashTable(len(animals), animals.dtype["class"])
    >>> enum = animal_classes.add(animals["class"])

    >>> RaggedArray.group_by(animals["name"], enum)
    RaggedArray.from_nested([
        ['cow', 'moose', 'whale'],
        ['centipede', 'spider'],
        ['robin', 'woodpecker'],
    ])
    >>> animal_classes.keys
    array(['mammal', 'insect', 'bird'], dtype='<U15')
