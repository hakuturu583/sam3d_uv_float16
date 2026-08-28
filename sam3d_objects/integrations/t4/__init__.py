"""Bridge between SAM 3D Objects and the T4 dataset (tier4/t4-devkit v0.8.0).

Import the sub-modules directly. They are layered so the coordinate math can be
used and tested without CUDA, ``torch`` or ``t4-devkit`` being installed, and
re-exporting the heavier ones here would defeat that:

* :mod:`~sam3d_objects.integrations.t4.frames` -- ``numpy`` frame conventions.
* :mod:`~sam3d_objects.integrations.t4.align`  -- ``numpy`` box alignment solver.
* :mod:`~sam3d_objects.integrations.t4.gaussian_ops` -- ``torch`` ops on ``Gaussian``.
* :mod:`~sam3d_objects.integrations.t4.dataset` -- ``t4-devkit`` loading helpers.
* :mod:`~sam3d_objects.integrations.t4.pipeline` -- end to end, needs both.
"""
