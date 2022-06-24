#!/bin/sh

for file in $1/*png; do
	fileName=$(basename "$file")
	mv -v $file $2/$3$fileName
done 
