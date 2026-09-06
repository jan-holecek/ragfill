#!/bin/bash
set -e

echo "Preparing directories..."
mkdir -p data/elasticsearch
mkdir -p data/mongodb

echo "Setting permissions..."
sudo chown -R 1000:1000 data/elasticsearch
sudo chown -R 1000:1000 data/mongodb

echo "Done"