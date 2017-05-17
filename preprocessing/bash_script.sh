#!/bin/bash

num_folders="$(aws s3 ls podcastsdatabase/dataset/ | wc -l)"
((num_folders-=2))
echo $num_folders

dfn="podcasts"
ufn="dataset"
d="aws s3 cp s3://podcastsdatabase/dataset/"
u="aws s3 cp $dfn/trans.mp3 s3://podcastsdatabase/dataset/"
g="python get_transition.py"
count=1

for i in $(seq 1 1 $num_folders)
do
	echo $i
	l="$(aws s3 ls podcastsdatabase/dataset/"$i"/ | awk -F" " '{$1=$2=$3=""; print $0}' | cut -c4-)"
	IFS=$'\n'
	for item in $l
	do
		echo "item: $item"
		if [[ $item == *".metadata"* ]]; then
			continue
		fi
		if [[ $item == "song"*"1.mp3"* ]]; then
			c1="$d$i/$item $dfn/s1.mp3"
		fi
		if [[ $item == "song"*"2.mp3"* ]]; then
			c1="$d$i/$item $dfn/s2.mp3"
		fi
		if [[ $item == *"transition"* ]]; then
			c1="$d$i/$item $dfn/music.mp3"
		fi
		
		#podcast download
		eval $c1
	
	
		#de-setup
	#	com="rm -r $dfn/*"
	#	eval $com
	#	com="rm -r $ufn/*"
	#	eval $com
	done

	#python
	c1="$g"
	eval $c1
	
	#upload dataset
	c1="$u$i/transition_part$i.mp3"
	eval $c1
done
