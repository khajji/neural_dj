from pydub import AudioSegment

def get_transition():
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
	
	song.export("podcasts/trans.mp3", format="mp3")

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

get_transition()
