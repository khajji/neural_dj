#!/bin/bash

min_number() {
    printf "%s\n" "$@" | sort -g | head -n1
}

num_folders="$(aws s3 ls podcastsdatabase/dataset/ | wc -l)"
((num_folders-=2))
echo $num_folders

dfn="podcasts"
ufn="dataset"
d="aws s3 cp s3://podcastsdatabase/dataset/"
u="aws s3 cp $ufn/ s3://podcastsdatabase/dataset/ --recursive"
g="python podcast_partitioner.py"
count=1

song1=1000000000
trans=1000000000

for i in $(seq 1 1 $num_folders)
do
	echo $i
	l="$(aws s3 ls podcastsdatabase/dataset/"$i"/ | awk -F" " '{$1=$2=$4=""; print $0}' | cut -c3-)"
	IFS=$'\n'
	num=0
	for item in $l
	do
		if [[ $num == 0 ]]; then
			((num++))
			continue
		fi
		if [[ $num == 1 ]]; then
			song1="$(min_number $item $song1)"
		fi
		if [[ $num == 2 ]]; then
			song1="$(min_number $item $song1)"
		fi
		if [[ $num == 3 ]]; then
			trans="$(min_number $item $trans)"
		fi
		((num++))
		#podcast download
		
	done
	echo "$i" >> data_files.txt
	echo "$song1" >> data_files.txt
	echo "$trans" >> data_files.txt
done
