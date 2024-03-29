
from setuptools import setup, find_packages

from codecs import open
from os import path
import re


## Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


## For single-sourceing the package version
def read(*parts):
    with open(path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")



setup(
    name='charge',
    version=find_version("charge", "version.py"),
    description='classical electrodynamics of charged particle',
    url='https://github.com/jam31118/charge',
    author='sahn',
    author_email='jam31118@gmail.com',
    classifiers=[
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='physics',
    packages=find_packages(),
    install_requires=['numpy','matplotlib','scipy'],
    long_description=long_description,
    license = 'GPLv3'
)

