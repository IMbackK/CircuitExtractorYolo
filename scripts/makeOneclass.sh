#!/bin/sh

for file in $1/*txt; do
	echo $file
	sed 's/^./0/g' "$file" > "$file.out"
	rm "$file"
	mv "$file.out" "$file"
done;
