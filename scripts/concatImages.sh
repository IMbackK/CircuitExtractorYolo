#!/bin/sh

montage -mode concatenate -frame 10 -mattecolor none -background white -tile 5x5 $@ result.png

for file in result*; do
    convert $file -fuzz 0% -fill 'rgb(255,255,255)' -opaque 'rgb(223,223,223)' $file
done
