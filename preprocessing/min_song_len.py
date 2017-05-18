import math

minsong = 1000000000
with open('song_lengths.txt') as fp:
	l = fp.read()
	l = l.split('\n')
	i = 0
	while(i < len(l)-2):
		minsong = min(minsong, int(l[i+1]))
		minsong = min(minsong, int(l[i+2]))
		i += 3
		
print(minsong)
