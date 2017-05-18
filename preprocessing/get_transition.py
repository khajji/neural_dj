from pydub import AudioSegment
import sys

def get_transition(s_num):
	comb = AudioSegment.from_mp3("podcasts/music.mp3")
	s1 = AudioSegment.from_mp3("podcasts/s1.mp3")
	s2 = AudioSegment.from_mp3("podcasts/s2.mp3")
	
	s1d = len(s1)
	s2d = len(s2)
	
	song = comb[s1d:-s2d]
	print(len(comb))
	print(len(s1))
	print(len(s2))
	print(len(song))

	saveToFile(s_num, s1d, s2d)
	song.export("podcasts/trans.mp3", format="mp3")

def saveToFile(s_num, s1d, s2d):
	f = open("song_lengths.txt", "a")
	f.write(str(s_num))
	f.write("\n")
	f.write(str(s1d))
	f.write("\n")
	f.write(str(s2d))
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
get_transition(s_num)
