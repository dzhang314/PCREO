# -*- coding: utf-8 -*-
"""mpfr_vector provides an arbitrary-precision mathematical vector class."""

# Python standard library imports
from collections.abc import Sequence

# Third-party imports
import gmpy2 # pylint: disable=import-error


class MPFRVector(Sequence):
    """MPFRVector represents an aribitrary-precision mathematical vector."""

    MPFR_TYPE = type(gmpy2.mpfr(0))

    def __init__(self, entries):
        self.entries = list(map(gmpy2.mpfr, entries))

    def __repr__(self):
        return "MPFRVector([" + ", ".join(map(repr, self.entries)) + "])"

    def __add__(self, other):
        if isinstance(other, MPFRVector):
            if len(self.entries) != len(other.entries):
                raise ValueError("cannot add MPFRVectors with different lengths")
            return MPFRVector(x + y for x, y in zip(self.entries, other.entries))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MPFRVector):
            if len(self.entries) != len(other.entries):
                raise ValueError("cannot subtract MPFRVectors with different lengths")
            return MPFRVector(x - y for x, y in zip(self.entries, other.entries))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MPFRVector.MPFR_TYPE):
            return MPFRVector(x * other for x in self.entries)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, MPFRVector.MPFR_TYPE):
            return MPFRVector(other * x for x in self.entries)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MPFRVector.MPFR_TYPE):
            return MPFRVector(x / other for x in self.entries)
        return NotImplemented

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.entries[index]
        if isinstance(index, slice):
            return MPFRVector(self.entries[index])
        raise TypeError("MPFRVector indices must be integers or slices")

    def dot(self, other):
        """Return the dot product of this MPFRVector with another MPFRVector."""
        if not isinstance(other, MPFRVector):
            raise ValueError("cannot take dot product of MPFRVector with non-MPFRVector")
        if len(self.entries) != len(other.entries):
            raise ValueError("cannot take dot product MPFRVectors with different lengths")
        return gmpy2.fsum(x * y for x, y in zip(self.entries, other.entries))

    def __matmul__(self, other):
        return self.dot(other)

    def norm_squared(self):
        """Return the squared Euclidean norm of this MPFRVector."""
        return gmpy2.fsum(map(gmpy2.square, self.entries))

    def norm(self):
        """Return the Euclidean norm of this MPFRVector."""
        return gmpy2.sqrt(self.norm_squared())

    def rec_norm(self):
        """Return the reciprocal of the Euclidean norm of this MPFRVector."""
        return gmpy2.rec_sqrt(self.norm_squared())

    def normalized(self):
        """Return a normalized copy of this MPFRVector."""
        r = self.rec_norm()
        return MPFRVector(x * r for x in self.entries)
