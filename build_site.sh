#!/bin/sh
# Puts the web page, the Python code and the models together in _site/ (what GitHub Pages serves).
set -e
rm -rf _site
mkdir -p _site
cp -r web/. _site/
cp checkpoint.py door.py _site/
cp -r models _site/models
touch _site/.nojekyll
