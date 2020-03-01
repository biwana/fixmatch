#!/bin/bash
sudo docker run --runtime nvidia --rm -it -v `pwd`:/work -w /work fixmatch "$@"
