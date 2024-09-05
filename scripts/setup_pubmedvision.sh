#!/bin/bash

for ((i=0; i<20; i++))
do
    unzip -j $HOME/data/pubmedvision/images_$i.zip -d $HOME/data/pubmedvision/images/ & # wait patiently, it takes a while...
done