from pydub import AudioSegment
from scipy import io
import numpy as np
import sys
import glob
import librosa
import os

# secs
trans_lens = [15] # [30, 60]
song_len = 30
chunks = 7.5

def downsample_and_store(s_num):
	global s1, s2, m1
	s1.frame_rate = 44100
	s2.frame_rate = 44100
	m1.frame_rate = 44100

	#save
	s1.export("podcasts/s1_"+ str(s_num) +".mp3", format="mp3")
	s2.export("podcasts/s2_"+ str(s_num) +".mp3", format="mp3")
	m1.export("podcasts/podcast_"+ str(s_num) +".mp3", format="mp3")

def get_real_transition(s_num):
	global s1, s2, m1
	s1d = len(s1)
	s2d = len(s2)
	m1d = len(m1)

	song = m1[s1d:-s2d]
	print(m1d)
	print(s1d)
	print(s2d)
	print(len(song))

	saveToFile1(s_num, s1d, s2d, len(song))
	song.export("podcasts/trans_"+ str(s_num) +".mp3", format="mp3")

def saveToFile1(s_num, s1d, s2d, t1d):
        f = open("song_lengths.txt", "a")
        f.write(str(s_num))
        f.write("\n")
        f.write(str(s1d))
        f.write("\n")
        f.write(str(s2d))
        f.write("\n")
        f.write(str(t1d))
        f.write("\n")
        f.close()

def get_training(s_num):
	global s1, s2, m1

	s1d = len(s1)
	s2d = len(s2)
	m1d = len(m1)
	mid = (s1d + m1d - s2d)/2

	real_trans_len = m1d - s1d - s2d
	saveToFile2(s_num, real_trans_len)

	for t in trans_lens:
		# label y
		trans_start = mid - t*1000/2
		song = m1[trans_start:trans_start+t*1000]
		print(len(song))
		song.export("podcasts/training_data/set_" + str(t) + "/music_data/transition" + ".mp3", format="mp3")
		
		tran_features = get_features("podcasts/training_data/set_" + str(t) + "/music_data/transition" + ".mp3")

		#dhex = song._data
		#ddec = map(ord, dhex)
		#dnp = np.array(ddec)
		tran_features = np.delete(tran_features, -1, 1)
		_, tt = tran_features.shape

		my_dict = {}
		my_dict['x'] = tran_features
		my_dict['y'] = 1.0
		io.savemat("podcasts/training_data/set_" + str(t) + "/binary_data/y" + ".mat", my_dict)
		
		#song1
		_s1 = m1[:trans_start]
		_s1d = len(_s1)
		diff = (_s1d - song_len*1000)/3
		song = _s1[:chunks*1000]
		song = song + _s1[chunks*1000 + diff:2*chunks*1000 + diff]
		song = song + _s1[2*chunks*1000 + 2*diff:3*chunks*1000 + 2*diff]
		song = song + _s1[3*chunks*1000 + 3*diff:4*chunks*1000 + 3*diff]
		print(len(song))
		song.export("podcasts/training_data/set_" + str(t) + "/music_data/s1" + ".mp3", format="mp3")
		
		#s1_features = get_features("podcasts/s1_"+ str(s_num) +".mp3")
		s1_features = get_features("podcasts/training_data/set_" + str(t) + "/music_data/s1" + ".mp3")

		#dhex = song._data
		#ddec = map(ord, dhex)
		#dnp = np.array(ddec)
		
		#song2
		_s2 = m1[trans_start+t*1000:]
		_s2d = len(_s2)
		diff = (_s2d - song_len*1000)/3
		song = _s2[:chunks*1000]
		song = song + _s2[chunks*1000 + diff:2*chunks*1000 + diff]
		song = song + _s2[2*chunks*1000 + 2*diff:3*chunks*1000 + 2*diff]
		song = song + _s2[3*chunks*1000 + 3*diff:4*chunks*1000 + 3*diff]
		print(len(song))
		song.export("podcasts/training_data/set_" + str(t) + "/music_data/s2" + ".mp3", format="mp3")
		
		s2_features = get_features("podcasts/training_data/set_" + str(t) + "/music_data/s2" + ".mp3")
		
		#dhex = song._data
		#ddec = map(ord, dhex)
		#dnp = np.append(dnp, np.array(ddec))

		# stitch features
		print(tran_features.shape)
		print(s1_features.shape)
		print(s2_features.shape)
		#print(dnp.shape)
		#features = np.append(dnp, s1_features)
		features = np.hstack([s1_features, s2_features])
		print(features.shape)

		my_dict = {}
		my_dict['x'] = features
		io.savemat("podcasts/training_data/set_" + str(t) + "/binary_data/x" + ".mat", my_dict)

def saveToFile2(s_num, l):
	f = open("real_trans_lengths.txt", "a")
	f.write(str(s_num))
	f.write("\n")
	f.write(str(l))
	f.write("\n")
	f.close()

def get_features(audio_path):
	#features
	y, sr = librosa.load(audio_path, sr=44100)

	#mel
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
	log_S = librosa.logamplitude(S, ref_power=np.max)
	#S_flat = S.flatten()

	# mel split
	y_harmonic, y_percussive = librosa.effects.hpss(y)
	S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
	S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

	log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
	log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)
	##log_Sh = log_Sh.flatten()
	##log_Sp = log_Sp.flatten()
	#Sh_flat = S_harmonic.flatten()
	#Sp_flat = S_percussive.flatten()

	S_split = np.vstack([log_Sh, log_Sp])
	features = np.vstack([log_S, S_split])

	#croma
	C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
	#C_flat = C.flatten()

	features = np.vstack([features, C])

	#mfcc
	mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)
	delta_mfcc  = librosa.feature.delta(mfcc)
	delta2_mfcc = librosa.feature.delta(mfcc, order=2)
	m = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
	#m_flat = m.flatten()

	features = np.vstack([features, m])

	#beats
	#tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
	#beats_flat = beats.flatten()

	#features = np.append(features, beats_flat)
	return m
	
# main
s_num = sys.argv[1]

s1 = AudioSegment.from_mp3("podcasts/song" + str(s_num) + "1.mp3")
s2 = AudioSegment.from_mp3("podcasts/song" + str(s_num) + "2.mp3")
m1 = AudioSegment.from_mp3("podcasts/transition" + str(s_num) + ".mp3")

downsample_and_store(s_num)
get_real_transition(s_num)
get_training(s_num)
