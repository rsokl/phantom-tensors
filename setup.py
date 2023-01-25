import versioneer
from setuptools import find_packages, setup

DISTNAME = "phantom_tensors"
LICENSE = "MIT"
AUTHOR = "Ryan Soklaski"
AUTHOR_EMAIL = "rsoklaski@gmail.com"

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = [
    "typing-extensions >= 4.1.0",
]
TESTS_REQUIRE = [
    "beartype >= 0.10.4",
    "pytest >= 3.8",
    "hypothesis >= 6.28.0",
]

DESCRIPTION = (
    "Tensor types with variadic shapes that support static and runtime type checking"
)
LONG_DESCRIPTION = """The goal of this project is to let users write tensor-like types with variadic shapes (PEP 646) that are amendable to both static type checking (without requiring a mypy plugin), as well as cross-tensor consistent runtime checking of shapes.

phantom-tensors is designed to be light weight and simple to use: we take a minimalistic approach to its API and its only dependency is `typing-extensions`.
"""


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    python_requires=">=3.7",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
    extras_require={
        "torch": ["torch>=1.7.0"],
        "numpy": ["numpy>=1.21.0"],
        "beartype": ["beartype >= 0.10.4"],
    },
    url="https://github.com/rsokl/phantom-tensors",
    download_url="https://github.com/rsokl/phantom-tensors/tarball/v"
    + versioneer.get_version(),
)
