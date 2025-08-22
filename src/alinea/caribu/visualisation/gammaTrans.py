from numpy import array

def gammaTrans(values, gamma=1,minval =None,maxval=None):
    """    return value normalised and raised at exponent gamma
    """
    values = array(values)

    if minval is None:
        m = values.min()
    else:
        m = minval

    if maxval is None:
        M = values.max()
    else:
        M = maxval

    if m == M:
        norm = 1.
    else:
        norm = M - m
    res = ((values - m) / float(norm))**gamma

    return res,
