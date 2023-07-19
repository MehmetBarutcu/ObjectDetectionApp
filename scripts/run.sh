#!/bin/sh
cd ..
python detect_speed.py --weights best.pt --source "C:\Users\mbarut\Desktop\car detection\vehicle-speed-counting\source" --interval 5 --no-trace --view-img --configfile "config.json" --device 0