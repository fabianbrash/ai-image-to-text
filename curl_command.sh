#!/bin/bash


curl -X POST -F "image=@/home/frjb-nix/workspace/python/ai-ml/image-to-text/dataset/train/images/yinger.png" http://localhost:8000/predict/
