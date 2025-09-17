""" Miscellaneous helper nodes used for building dataflows
"""
from functools import reduce




def reduceDict(dictlist):
    """ Bind dict that share the same keys and whose values are lists
    """

    def _bind(d, e):
        vals = []
        for k in d:
            if isinstance(d[k], list):
                nl = d[k]
            else:
                nl = [d[k]]
            nl.append(e[k])
            vals.append(nl)
        return dict(list(zip(list(d.keys()), vals)))

    reduced = reduce(_bind, dictlist)

    return reduced


def filterby(indices, values, condition):
    """
    Return values whose indices match condition
    """
    index_value = filter(lambda x: condition(x[0]), zip(indices, values))
    res = [value for index, value in index_value]

    return res


def mydict(list_of_tuple):
    """ create a dict from a list of tuples
    """
    d = dict(list_of_tuple)
    return d