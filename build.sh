#!/usr/bin/env bash
# Exit on error
set -o errexit

# Upgrade pip and critical build tools using python -m pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies with no-cache to save space
python -m pip install -r requirements.txt --no-cache-dir

python manage.py collectstatic --no-input
python manage.py migrate
