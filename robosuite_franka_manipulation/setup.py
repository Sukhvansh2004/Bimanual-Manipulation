from setuptools import setup, find_packages

setup(
    name="robosuite_franka_handoff",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "robosuite",
        "numpy",
        "scipy",
    ],
    extras_require={
        "graspgen": ["graspgen"],
    },
)