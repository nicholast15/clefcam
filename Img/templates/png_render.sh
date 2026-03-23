#!/bin/bash

SIZES=(16 32 64 128)

for svg_file in *.svg; do
	filename=$(basename "$svg_file" .svg)

	for size in "${SIZES[@]}"; do
		convert "$svg_file" -resize "${size}x${size}" \
            	-quality 500 -strip \
            	"${filename}_${size}.png"
	done
	echo "Processed $filename"
done
