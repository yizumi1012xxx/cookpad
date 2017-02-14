#!/bin/bash

rsync -av --delete --exclude='data' --exclude='result' --exclude='*.pyc' --exclude='nohup.out' -e "ssh -i~/.ssh/pem/cookpad.pem -p2222" ~/Documents/workspace/cookpad/ challenge88@203.137.180.177:~/cookpad

# { nohup python src/main.py; } > ../resnet152.out &
