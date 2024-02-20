#!/usr/bin/bash

curl -L https://www.cs.purdue.edu/homes/jsetpal/data/animal-10.tgz -o data.tgz
tar xvzf data.tgz
rm data.tgz
mkdir models
