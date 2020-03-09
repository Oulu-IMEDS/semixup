#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = (
    'numpy', 'opencv-python', 'Pillow==6.1', 'matplotlib', 'solt==0.1.8', 'torch==1.2.0', 'torchvision==0.4.1', 'tqdm', 'scikit-learn==0.20.4', 'pandas==0.24.1', 'collagen',
    'tensorboardX', 'sas7bdat', 'dill', 'pyyaml')

setup_requirements = ()

test_requirements = ('pytest',)

description = """Semixup: In- and Out-of-Manifold Regularization for Deep Semi-Supervised Knee Osteoarthritis Severity Grading from Plain Radiographs
"""

setup(
    author="Huy Hoang Nguyen",
    author_email='huy.nguyen@oulu.fi',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux'
    ],
    description=description,
    install_requires=requirements,
    license="MIT license",
    long_description=description,
    include_package_data=True,
    keywords='Deep learning, Semi-supervised learning, In- and out-of-manifold regularization',
    name='semixup',
    packages=find_packages(include=['semixup', 'pimodel', 'mixmatch', 'ict', 'sl']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MIPT-Oulu/semixup',
    version='0.0.1',
    zip_safe=False,
)
