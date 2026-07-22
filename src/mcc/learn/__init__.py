"""A learning lab for PyTorch + CUDA.

These are *inspection* helpers -- they don't build models for you, they let you SEE
what your own code is doing under the hood at each step. Pair them with GUIDE.md.

Typical use in a REPL / notebook:

    from mcc.learn import inspect as I
    from mcc.learn import viz

    t = ...            # a tensor YOU created
    I.describe_tensor(t, "t")
    I.show_storage(t)

Requires torch (install separately, see README).
"""
