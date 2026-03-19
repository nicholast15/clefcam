#!/bin/bash
#usage: 'source env_init.sh' in a Bash shell
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

