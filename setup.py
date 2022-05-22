# -*- coding: utf-8 -*-
"""
"""

from setuptools import setup, find_packages
import runpy
from pathlib import Path

from cslug.building import bdist_wheel, copy_requirements, build_slugs, \
    CSLUG_SUFFIX

HERE = Path(__file__).resolve().parent

readme = (HERE / 'README.rst').read_text("utf-8")

setup(
    author="Brénainn Woodsend",
    author_email='bwoodsend@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Ragged (rows with different lengths) 2D NumPy arrays.",
    install_requires=copy_requirements(),
    entry_points={"pyinstaller40": "hook-dirs=rockhopper:_PyInstaller_hook"},
    extras_require={
        "test": [
            'pytest>=3', 'pytest-order', 'coverage', 'pytest-cov',
            'coverage-conditional-plugin', 'hypothesis'
        ]
    },
    license="MIT license",
    long_description=readme,
    keywords='rockhopper',
    name='rockhopper',
    packages=find_packages(include=['rockhopper', 'rockhopper.*']),
    url='https://github.com/bwoodsend/rockhopper',
    version=runpy.run_path(HERE / "rockhopper/_version.py")["__version__"],
    zip_safe=False,
    package_data={"rockhopper": ["*/*" + CSLUG_SUFFIX, "*/*.json"]},
    cmdclass={
        "build": build_slugs("rockhopper._ragged_array:slug"),
        "bdist_wheel": bdist_wheel,
    },
)
