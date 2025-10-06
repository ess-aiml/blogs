#!/bin/bash
set -e

# Build the site
hugo

# Go to public folder and push to gh-pages branch
cd public
git init
git add .
git commit -m "Deploy Hugo site"
git push --force https://github.com/levuvietphong/ess-ai-ml.git main:gh-pages
cd ..
