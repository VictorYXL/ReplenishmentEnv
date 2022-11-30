from setuptools import find_packages
from setuptools import setup
setup(
    name="replenishment",
    version="1.0",
    python_requires=">=3.8",
    install_requires=[
        "gym>=0.26.0",
        "pyaml",
        "pandas",
        "numpy>=1.20.3"
    ],
    packages=find_packages(exclude=["Example"]),
    include_package_data=True,
    package_data={
        "ReplenishmentEnv": ["config/*.yml", "data/*/*", "ReplenishmentEnv.*py"],
    }
)
