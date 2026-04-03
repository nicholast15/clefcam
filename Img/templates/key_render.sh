#!/bin/bash

SIZES=(8 16 32)

for svg_file in sharp.svg; do
	filename=$(basename "$svg_file" .svg)

	for size in "${SIZES[@]}"; do
		convert "$svg_file" -resize "${size}x${size}" \
            	-quality 500 -strip \
            	"${filename}_${size}.png"
	done
	echo "Processed $filename"
done

for svg_file in flat.svg; do
	filename=$(basename "$svg_file" .svg)

	for size in "${SIZES[@]}"; do
		convert "$svg_file" -resize "${size}x${size}" \
            	-quality 500 -strip \
            	"${filename}_${size}.png"
	done
	echo "Processed $filename"
done
