"""Attribute-level stand-in for the compiled ``pointnet2_ops._ext`` module.

Importing succeeds; touching any operator fails loudly.
"""


def __getattr__(name: str):
    raise RuntimeError(
        f"pointnet2_ops._ext.{name} was called, but this is the CPU-only CI stub "
        "(see .github/ci/stubs/pointnet2_ops/__init__.py). Real PointNet2 "
        "operators need the CUDA extension built by install.sh on a GPU machine."
    )
