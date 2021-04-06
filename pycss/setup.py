from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        f.read()


setup(
    name="pycss",
    version="0.1",
    description="Generalized parameter estimation in ME GRE chemical species separation",
    long_description=readme(),
    url="https://github.com/maxdiefenbach/MR_CSS",
    author="Maximilian Diefenbach",
    packages=find_packages(),
    install_requires=["numpy", "numba", "matplotlib", "scipy", "pandas"],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)
