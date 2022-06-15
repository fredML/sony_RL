#!/bin/bash
#export PYTHONPATH=/usr/lib/python3.8
# Launch blender python script (blender application must be defined in your PATH system)
/mnt/diskSustainability/frederic/blender-2.93.9-linux-x64/blender -E CYCLES -b -noaudio -P vscanner_blender.py
