#!/bin/sh

for file in $1/*txt; do
	if ! [[ "$file" == *"classes.txt" ]]; then
		echo $file
		sed 's/^./0/g' "$file" > "$file.out"
		rm "$file"
		mv "$file.out" "$file"
	fi
done;
