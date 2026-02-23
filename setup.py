from setuptools import setup, find_packages

setup(
    name="s4_lib",
    version="0.1.0",
    description="S4 & S4D: Structured State Spaces for Sequence Modeling",
    author="Meggison Oritsemisan",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy>=1.21",
    ],
    extras_require={
        "tutorials": ["matplotlib>=3.4"],
        "dev": ["pytest>=7.0", "matplotlib>=3.4"],
    },
)
