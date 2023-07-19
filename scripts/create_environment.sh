#!/bin/sh
conda create --name vehicle-counting
conda activate vehicle-counting
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install --upgrade pip
pip install -r requirements.txt

