"""Unit test package for rockhopper."""


class _Slice(object):

    def __getitem__(self, item):
        return item


sliced = _Slice()
