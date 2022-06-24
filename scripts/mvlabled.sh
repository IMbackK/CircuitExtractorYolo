#!/bin/sh

pngExt="png"

for file in $1/*txt; do
	pngFileName=$(basename "$file" txt)$pngExt
	echo $1/$pngFileName to $2
	mv "$1/$pngFileName" "$2"
	mv "$file" "$2"
done;
