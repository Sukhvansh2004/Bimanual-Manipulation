"""CPU-only import shim for GraspGen's PointNet2 CUDA extension.

``grasp_gen.models.pointnet.pointnet2_utils`` does::

    try:
        from pointnet2_ops import _ext as ops
    except ImportError:
        ops = torch.utils.cpp_extension.load(..., with_cuda=True)

The real package is a compiled CUDA extension, and the fallback JIT path also
requires ``nvcc``. Neither is available on a GPU-less CI runner, so without a
shim the import fails and *every* test module dies at collection —
``bimanual_manipulation`` imports ``grasp_gen.grasp_server`` transitively via
``perception/grasp_predictor.py``.

The test suite never invokes a PointNet2 kernel (it covers transforms, planning,
IK, cost/DP assignment and service plumbing, mocking model inference), so
satisfying the import is enough. Any attempt to actually *call* into the
extension raises a clear error rather than silently returning wrong numbers.

This package is only ever on ``PYTHONPATH`` inside CI; a real GPU environment
built by ``install.sh`` shadows it with the compiled extension.
"""

from . import _ext  # noqa: F401

__all__ = ["_ext"]
