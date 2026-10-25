"""Caribu dataflow Tests"""

run_test = True
try:
    from openalea.core.alea import run, function
except ImportError:
    run_test = False

if run_test and False:

    def test_caribu():
        """ Test Tutorial LIE """

        res = run(('openalea.caribu.demos', 'Tutorial'),
                  inputs={}, vtx_id=11)
        efficiency = res[0]
        if efficiency:
            assert 0.62 < res[0] < 0.63, res
