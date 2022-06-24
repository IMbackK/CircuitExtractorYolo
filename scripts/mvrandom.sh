#!/bin/sh

sample=`ls $1/*txt | sort -R | head -n 10`

while IFS= read -r line; do
    file=`basename -s .txt "$line"`
    mv "$1/$file".png"" $2
    mv "$1/$file".txt"" $2
done <<< "$sample"

