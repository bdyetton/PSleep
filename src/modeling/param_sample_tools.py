import numpy as np


def rand_trans_0back(nshape):
    tp = np.random.dirichlet([1]*nshape)
    assert np.sum(tp) - 1 < 0.00000001
    return tp


def rand_trans_1back(nshape):
    tp = np.random.dirichlet([1]*nshape, nshape)
    for row in range(nshape):
        assert np.sum(tp[row, :]) - 1 < 0.00000001
    return tp


def rand_trans_1back_tdep(nsamples, nshape):
    b0 = np.abs(np.random.normal(0.5, 0.3, size=(1, nshape, nshape)))
    b0 += np.identity(nshape)
    b0 = b0/np.sum(b0, axis=1)
    b1 = np.random.normal(0.01, 0.01, size=(1, nshape, nshape))  # FIXME not sure about this...
    t = np.moveaxis(np.arange(nsamples-1)[np.newaxis][np.newaxis], 2, 0)
    tps = b0+b1*t
    tps = tps/np.expand_dims(np.sum(tps, axis=2),-1)
    return tps, b0, b1


def rand_trans_2back(nshape):
    stacker = []
    for _ in range(nshape):
        stacker.append(np.random.dirichlet([1]*nshape, nshape))
    tp = np.stack(stacker, axis=0)
    for row in range(nshape):
        for col in range(nshape):
            assert np.sum(tp[row, col, :]) - 1 < 0.00000001
    return tp