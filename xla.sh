#!/bin/bash

curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

python pytorch-xla-env-setup.py --version 1.7
