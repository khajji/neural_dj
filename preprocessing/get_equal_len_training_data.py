from pydub import AudioSegment
import sys

# secs
trans_lens = [15, 30, 60]
song_len = 100
chunks = 25

def get_training(s_num):
	comb = AudioSegment.from_mp3("podcasts/music.mp3")
	s1 = AudioSegment.from_mp3("podcasts/s1.mp3")
	s2 = AudioSegment.from_mp3("podcasts/s2.mp3")
	
	s1d = len(s1)
	s2d = len(s2)
	mid = (s1d + len(comb) - s2d)/2

	real_trans_len = len(comb) - s1d - s2d
	saveToFile(s_num, real_trans_len)
	
	for t in trans_lens:	
		trans_start = mid - t*1000/2
		song = comb[trans_start:trans_start+t*1000]
		print(len(song))	
		song.export("podcasts/training_transitions/" + str(t) + ".mp3", format="mp3")
	
	#song1
	diff = (s1d - 100*1000)/3
	song = s1[:chunks*1000]
	song = song + s1[chunks*1000 + diff:2*chunks*1000 + diff]
	song = song + s1[2*chunks*1000 + 2*diff:3*chunks*1000 + 2*diff]
	song = song + s1[3*chunks*1000 + 3*diff:4*chunks*1000 + 3*diff]
	print(len(song))
	song.export("podcasts/training_songs/s1" + ".mp3", format="mp3")

	#song2
	diff = (s2d - 100*1000)/3
	song = s2[:chunks*1000]
	song = song + s2[chunks*1000 + diff:2*chunks*1000 + diff]
	song = song + s2[2*chunks*1000 + 2*diff:3*chunks*1000 + 2*diff]
	song = song + s2[3*chunks*1000 + 3*diff:4*chunks*1000 + 3*diff]
	print(len(song))
	song.export("podcasts/training_songs/s2" + ".mp3", format="mp3")

def saveToFile(s_num, l):
	f = open("real_trans_lengths.txt", "a")
	f.write(str(s_num))
	f.write("\n")
	f.write(str(l))
	f.write("\n")
	
	f.close()

def reduce_length():
	s1 = AudioSegment.from_mp3("podcasts/s1.mp3")
	song = s1[:1]
	for i in range(len(s1)/2):
		print(i)
		if (i==0):
			continue
		song = song + s1[i*2:i*2+1]
	
	print(len(s1))
	print(len(song))
	song.export("podcasts/rl.mp3", format="mp3")

s_num = sys.argv[1]
get_training(s_num)
