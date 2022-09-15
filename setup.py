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
    "phantom-types >= 0.17.1",
    "typing-extensions >= 4.1.0",
]
TESTS_REQUIRE = [
    "beartype >= 0.10.4",
    "pytest >= 3.8",
    "hypothesis >= 6.28.0",
]

DESCRIPTION = "Configurable, reproducible, and scalable workflows in Python, via Hydra"
LONG_DESCRIPTION = """The goal of this project is to let users write tensor-like types with variadic shapes that are amendable to both detailed static type checking as well as cross-tensor consistenct runtime type checking of shapes.
"""


setup(
    name=DISTNAME,
    version="0.0.0",
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
    },
)
