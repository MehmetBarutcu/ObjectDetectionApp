#!/bin/sh
sudo apt-get purge nvidia-*

sudo apt-get update

sudo apt-get autoremove

sudo apt install nvidia-driver-470