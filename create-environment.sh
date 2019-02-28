#!/bin/bash

HERE=$(realpath $(dirname $))

# TODO which apt packages do we need?

# -- Check for TORCH
TORCH=$(which th)
if [ -z $TORCH ]
then
	echo "Please install torch."
	echo "e.g. sudo apt install torch-trepl"
	echo "or from http://torch.ch/docs/getting-started.html#"
	echo "and make sure that 'th' is on your path."
	exit 1
fi

# --- Torch dependencies
luarocks install dpnn

# -- Python3 virtual environment
VENV="$HERE/venv"

rm -rf $VENV
python3 -m venv $VENV
source $VENV/bin/activate

cd $HERE
pip install wheel
pip install -r requirements.txt

if [ ! -d $HERE/models/dlib ] || [ ! -d $HERE/models/openface ]
then
	echo "Fetching models"
	cd $HERE/models
	./get-models.sh
	cd ..
fi


echo "Checking installation by running a basic unit test"
cd $HERE/tests
nosetests --verbose --nocapture openface_api_tests.py
