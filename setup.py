#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='cellmates',
    version='0.1.0',
    author='Vittorio Zampinetti, Harald Melin, Marzie Abdolhamdi',
    author_email='vz@kth.se, haralme@kth.se, marziea@kth.se',
    description='Cellmates is a maximum likelihood method for copy-number based single-cell tree reconstruction in'
                ' tumor data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Lagergren-Lab/cellmates',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.10',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        'console_scripts': [
            'cellmates=main:main',
        ],
    },
    extras_require={
        "dev": [
            "pytest>=8.0",
        ],
    },
)
