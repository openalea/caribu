def assert_almost_equal(a, b, places=7, msg=None):
    """
    Fail if the two objects are unequal as determined by their
    difference rounded to the given number of decimal places
    and comparing to zero.
    Note that decimal places (from zero) are usually not the same
    as significant digits (measured from the most signficant digit).
    See the builtin round() function for places parameter.
    """
    if msg is None:
        assert round(abs(b - a), places) == 0
    else:
        assert round(abs(b - a), places) == 0, msg
