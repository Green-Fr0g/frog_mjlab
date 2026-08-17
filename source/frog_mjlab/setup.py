"""Installation script for the frog_mjlab package."""

from setuptools import setup, find_packages

setup(
    name="frog-mjlab",
    version="0.1.0",
    packages=find_packages(include=["frog_mjlab", "frog_mjlab.*"]),
    install_requires=[
        "mjlab",
        "mujoco-warp",
        "warp-lang",
        "scipy",
    ],
)
