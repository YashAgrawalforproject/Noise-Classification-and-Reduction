import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import librosa
from numpy import genfromtxt
from keras.models import load_model
from noisereduce.noisereducev1 import reduce_noise
import soundfile as sf
# from tensorflow.python.training.saving.saveable_object_util import saveable_objects_from_trackable
# removal function

def denoise(data, pred):
	noise, sr2 = librosa.load(pred)
	reduced_noise = reduce_noise(audio_clip=data, noise_clip=noise, verbose=False)
	print(reduced_noise)
	# librosa.output.write_wav('clean.wav', reduced_noise, sr)
	sf.write('clear.wav', reduced_noise, sr)

model = load_model('model.h5')
print('\n\n\n Model Loaded \n\n\n')

# preprocessing using entire feature set
x_test = []
print('\nvariable created')
filename = str(input('Enter path to file: '))
print('\nfile given')
y, sr = librosa.load(filename)
mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=36).T, axis=0)
melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36, fmax=8000).T, axis=0)
chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=36).T, axis=0)
chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=36).T, axis=0)
chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=36).T, axis=0)
features = np.reshape(np.vstack((mfcc, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (36, 5))
x_test.append(features)
# y_test.append(label)

print('Length of Data: ', len(x_test))
x_test = np.array(x_test)
# y_test=np.array(y_test)
print('\n Test_array shape: ', x_test.shape)


x_test = np.reshape(x_test, (x_test.shape[0], 36, 5, 1))
print('\nFinal shape: ', x_test.shape)
ans = model.predict(x_test)
print(ans)

print('Class 0: Windy \n Class 1: Horn\n Class 2: Children-noise \n Class 3: Dog Bark \n Class 4: Drilling \n Class 5: Engine Idling\n Class 6: Gun Shot \n Class 7: Jackhammer\n Class 8: Siren \n Class 9: Street music\n')

#my_dict={0: 'Windy' , 1: 'Horn' ,  2: 'Dog Bark' , 3: 'Drilling'  ,4: 'Engine Idling',5: 'Gun Shot',  6:' Jackhammer',  7: 'Siren' , 8: 'Street music',}
my_dict={0: 'Windy' , 1: 'Horn', 2: 'Children-noise' ,  3: 'Dog Bark' ,  4: 'Drilling'  ,5: 'Engine Idling',6: 'Gun Shot',  7:' Jackhammer',  8: 'Siren' , 9: 'Street music'}




print(ans[0])
import copy
x = copy.copy(ans[0])

x = list(x)
print(x)
arr = []
ls = []
ls = list(ans[0])
print(ls)
while ( len(x) > 6):

	aud = max(x)
	index = ls.index(aud)
	# if(index == 2):
	# 	continue;

	x.remove(aud)
	arr.append(index)
# print(arr)
print('Resulted Index: ', arr)
print('\nNoises Present: ')
print('')
for idx in arr:
	print(idx)
	print(my_dict[idx])



##################################################


import IPython
from scipy.io import wavfile
from noisereduce.generate_noise import band_limited_noise
import numpy as np
import io

import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display

"""### Test on Customized Audio"""

source=filename
data, sr1 = librosa.load(source)



for i in arr:


	if i == 0:
		pred = "noise/ac1.wav"
		denoise(data, pred)
		pred = "noise/ac2.wav"
		denoise(data, pred)
	elif i == 1:
		pred = "noise/horn1.wav"
		denoise(data, pred)
		pred = "noise/horn2.wav"
		denoise(data, pred)
	elif i == 2:
		pred = "noise/children1.wav"
		denoise(data, pred)
		pred = "noise/children2.wav"
		denoise(data, pred)
	elif i == 3:
		pred = "noise/bark1.wav"
		denoise(data, pred)
		pred="noise/bark2.wav"
		denoise(data, pred)
	elif i == 4:
		pred = "noise/drill1.wav"
		denoise(data, pred)
		print("\n done part 1")
		pred = "noise/drill2.wav"
		denoise(data, pred)
		print("\n done part 2")
	elif i == 5:
		pred = "noise/engine1.wav"
		denoise(data, pred)
		pred = "noise/engine2.wav"
		denoise(data, pred)
	# elif i == 6:
	# 	pred = "noise/drill1.wav"
	# 	denoise(data, pred)
	# 	pred = "noise/drill2.wav"
	# 	denoise(data, pred)
	elif i == 7:
		pred = "noise/jack1.wav"
		denoise(data, pred)
		pred = "noise/jack2.wav"
		denoise(data, pred)
	elif i == 8:
		pred = "noise/siren1.wav"
		denoise(data, pred)
		pred = "noise/siren2.wav"
		denoise(data, pred)
	elif i == 9:
		pred = "noise/street1.wav"
		denoise(data, pred)
		pred = "noise/street2.wav"
		denoise(data, pred)
	else:
		print('Nothing there!')
print("\nCleaned Audio saved as: clear.wav")
""" ### Noise will remove """