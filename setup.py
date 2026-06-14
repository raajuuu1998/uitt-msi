from setuptools import setup, find_packages

setup(
    name="uitt",
    version="1.0.0",
    description="Universal Immune-Tumor Topology for cross-cancer MSI prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0", "numpy>=1.24", "pandas>=2.0",
        "scikit-learn>=1.3", "scipy>=1.10", "matplotlib>=3.7",
    ],
)
