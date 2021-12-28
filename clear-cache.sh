#!/bin/sh
#
# Delete Python's compiled *.pyc and __pycache__ files like a pro
# https://gist.github.com/jakubroztocil/7892597
#
# Usage:
# Delele *.pyc and __pycache__ files recursively in the current directory:
# $ pyc  
# 
# The same, but under /path:
# $ pyc /path
#    

if [ "$1" ]; then
    WHERE="$1"
else
    WHERE="$PWD"
fi

find "$WHERE" \
    -name '__pycache__' -delete -print \
    -o \
    -name '*.pyc' -delete -print