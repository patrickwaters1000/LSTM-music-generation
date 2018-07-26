import numpy as np
import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import pickle
import sys, os
import keras.backend as K
from PrepareData import *

options = sys.argv[1:]

look_back = 16
nbr_notes = 100

def get_some_data(name):
	data_top = pickle.load(open("data/pickled/{}TopVoice.p".format(name),"rb"))
	data_others = pickle.load(open("data/pickled/{}OtherVoices.p".format(name),"rb"))
	l = min(len(data_top),len(data_others))
	data_top = data_top[:l]
	data_others = data_others[:l]
	X,Y = [],[]
	for i in range(l-look_back):
		x_window = data_top[i:i+look_back]
		x = [[1 if v in notes else 0 for v in range(100)] for notes in x_window]
		y = [1 if v in data_others[i+look_back] else 0 for v in range(100)]
		if sum(y)==1:
			X.append(x)
			Y.append(y)
	return X,Y

X_train, Y_train = [], []
X_test, Y_test = [], []

X,Y = get_some_data("Fugue3")
X_train.extend(X)
Y_train.extend(Y)

X,Y = get_some_data("Fugue2")
X_test.extend(X)
Y_test.extend(Y)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print("Shapes {}, {}".format(X_train.shape,Y_train.shape))


opt1 = keras.optimizers.Adam(lr=1e-3)

def my_loss_function(y_true,y_pred):
	loss1 = keras.losses.mean_squared_error(y_true,y_pred)
	loss2 = K.mean(y_true*(y_true-y_pred)**2)
	return 4.0*loss2+loss1
keras.losses.my_loss_function = my_loss_function

if "-l" in options:
	M = keras.models.load_model("LSTM_model.h5")
	#K.set_value(opt1.lr,0.0)
	#M.optimizer=keras.optimizers.SGD(lr=0.0)
else:
	M = Sequential()
	M.add(LSTM(50,input_shape=(look_back,nbr_notes),return_sequences=True))
	M.add(LSTM(50))
	M.add(Dense(nbr_notes,activation="softmax"))
	M.compile(loss = my_loss_function, optimizer = opt1, metrics = ["acc"])

if "-t" in options:
	M.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=1000,batch_size=100)
	if "s" == input("s to save"):
		M.save("LSTM_model.h5")


twinkle_twinkle_little_star = 	[[48],[],[48],[],[55],[],[55],[],[57],[],[57],[],[55],[],[],[],[53],[],[53],[],[52],[],[52],[],[50],[],[50],[],[48],[],[],[],
	[55],[],[55],[],[53],[],[53],[],[52],[],[52],[],[50],[],[],[],[55],[],[55],[],[53],[],[53],[],[52],[],[52],[],[50],[],[],[],
	[48],[],[48],[],[55],[],[55],[],[57],[],[57],[],[55],[],[],[],[53],[],[53],[],[52],[],[52],[],[50],[],[50],[],[48],[],[],[]]


three_blind_mice = [
[52],[],[],[50],[],[],[48],[],[],[],[],[],
[52],[],[],[50],[],[],[48],[],[],[],[],[],
[55],[],[],[53],[],[53],[52],[],[],[],[],[],
[55],[],[],[53],[],[53],[52],[],[],[],[],[55],
[60],[],[60],[59],[57],[59],[60],[],[55],[55],[],[55],
[60],[],[60],[59],[57],[59],[60],[],[55],[55],[],[53],
[52],[],[],[50],[],[],[48],[],[],[],[],[],
[52],[],[],[50],[],[],[48],[],[],[],[],[]]
three_blind_mice = [[note+24 for note in step] for step in three_blind_mice]


def generate_voice(other_voices):
	l = len(other_voices)
	Y = [[] for i in range(look_back)]
	for i in range(l-look_back):
		x_window = other_voices[i:i+look_back]
		x = [[1 if v in notes else 0 for v in range(100)] for notes in x_window]
		x = np.array([x])
		pred = M.predict(x)[0]
		j = np.argmax(pred)
		Y.append([j])
	return Y



voice = generate_voice(three_blind_mice)
voices_together = [s+v for s,v in zip(three_blind_mice,voice)]
print(voices_together)
pickle.dump(voices_together,open("data/pickled/three_blind_mice.p","wb"))

dt = 60
time, events = 0, []
for step in voices_together:
	for note in step:
		events.append(Event("on",time,note))
		events.append(Event("off",time+dt,note))
	time += dt

events.sort(key=lambda x: x.time)
s = Song([Track(events)])
s.write("data/midi_outputs/three_blind_mice.mid")



