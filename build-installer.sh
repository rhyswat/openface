#!/bin/bash

rm -rf build dist MANIFEST openface.egg-info

python setup.py sdist bdist_wheel
