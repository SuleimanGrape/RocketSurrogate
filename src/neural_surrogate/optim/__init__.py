"""Gradient-based design optimization support for the neural surrogate.

This package makes the *whole* surrogate pipeline — raw design parameters →
engineered features → scaling → network → un-scaling → natural-unit targets —
differentiable end-to-end in torch, so an optimizer can follow
d(target)/d(raw design input). The numpy/pandas feature engineering in
gbt/preprocess.py is exact but opaque to autograd; diff_features re-implements
the identical formulas in torch (verified to match to machine precision).
"""
