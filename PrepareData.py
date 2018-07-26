'''
This script contains tools for reading and writing midi files.

I use the following object oriented design to represent the data in a midi file:
Song objects have a tracks attribute, which is a list of Track objects.
Track objects have an events attributte, which is a list of Event objects.
Event objects represent instructions like "note 60 is turned on at time 720".

Given a midi file, you can build a Song object.
This is done by constructing a MidiData object, and then using its .to_song() method.

Once you have a Song object, you can do two things:
1)	Use the .to_data() method to get a list of notes struck at each time step.
	After converting the list at each time step to categorical format, this can be fed into a neural network.
2)	Use the .write(fname) method to convert the Song object to midi format and write it to the file fname.
'''
from binascii import unhexlify
import sys
from operator import concat
from functools import reduce


hex_vals = {str(i):i for i in range(10)}
for k,v in zip(list("abcdef"),range(10,16)):
	hex_vals[k] = v
hex_chars = [str(i) for i in range(10)]+list("abcdef")

def hex_to_int(h):
	"interprets a string of hex digits as an integer"
	x=0
	for c in h:
		x = 16*x + hex_vals[c]
	return x

def int_to_hex(x,m):
	y = ''
	for i in range(m):
		q,x = divmod(x,16**(m-i-1))
		y+=hex_chars[q]
	return y

def list_to_number(L):
	"interprets a list of values 0-255 as an integer in base 256"
	m,x = 1,0
	while L:
		x += L.pop()*m
		m *= 256
	return x

def int_to_varlen(x):
	if x==0:
		return "00"
	s = ''
	while x>0:
		x,r = divmod(x,128)
		if len(s)>0:
			r += 128
		s=int_to_hex(r,2)+s
	return s

class Event:
	headers={"on":"90", "off":"80"}

	def __init__(self,event_type,time,note):
		self.event_type = event_type
		self.time = time
		self.note = note
	
	def repr(self):
		return (self.event_type, self.time, self.note)

	def to_str(self,prev_time):
		dt = int_to_varlen(self.time - prev_time)
		h = Event.headers[self.event_type]
		n = int_to_hex(self.note,2)
		v = "40"
		return dt+h+n+v, self.time

class Track:
	def __init__(self,events):
		self.events = events

	def repr(self):
		return [e.repr() for e in self.events]

	def to_midi(self):
		track_str = ""
		t = 0
		for e in self.events:
			chunk, t = e.to_str(t)
			track_str += chunk
		l = len(track_str) // 2	+ 4
		track_header = '4d54726b' +int_to_hex(l,8)
		track_end = '00ff2f00'
		return track_header + track_str + track_end


def rounded_quotient(x,m):
	q,r = divmod(x,m)
	d = int(r >= m/2)
	return q+d

class Song:
	def __init__(self,tracks):
		self.tracks = [x for x in tracks if x.events]

	def to_midi(self):
		track_strings = [t.to_midi() for t in self.tracks]
		nbr_tracks = len(track_strings)
		header = '4d546864'+int_to_hex(6,8)+'0001'+int_to_hex(nbr_tracks,4)+'0080'
		return header + reduce(concat, track_strings)

	def to_data(self,dt):
		"returns a list of the notes struck at each time step"
		iMax = max(rounded_quotient(T.events[-1].time,dt) for T in self.tracks)
		data = [[] for i in range(iMax)]
		for T in self.tracks:
			for e in T.events:
				if e.event_type=="on":
					i = rounded_quotient(e.time,dt)
					data[i].append(e.note)
		return data
		

	def write(self,fname):
		hex_str = self.to_midi()
		midi_bytes = unhexlify(hex_str)
		open(fname,"wb").write(midi_bytes)


class MidiData:
	def __init__(self,fname):
		data = open(fname,"rb").read()
		L = [int(x) for x in data]
		assert L[:8]==[77,84,104,100,0,0,0,6]
		self.file_format = list_to_number(L[8:10])
		self.nbr_tracks = list_to_number(L[10:12])
		self.division = list_to_number(L[12:14])
		print("File {} read.  It is type {} and has {} tracks.".format(fname,self.file_format,self.nbr_tracks))
		L = L[14:]
		self.tracks = []
		while L:
			assert L[:4]==[77, 84, 114, 107]
			track_len = list_to_number(L[4:8])
			X,L=L[8:8+track_len],L[8+track_len:]
			self.tracks.append( TrackData(X) )
		
	def to_song(self):
		tracks = [T.read() for T in self.tracks]
		return Song(tracks)
		
class TrackData:	
	def __init__(self,L):
		self.data = L
		self.t = 0
	
	def repr(self):
		return [int_to_hex(x,2) for x in self.data]

	def get_varlen(self):
		"""	given a list of ints beginning with a midi-var len number,
		pop that number from the beginning of L and return it 	"""
		x, keep_going = 0, 1
		while keep_going:
			a = self.data.pop(0)
			keep_going = a//128
			x = 128*x + (a%128)
		return x

	def get_event(self):
		dt = self.get_varlen()
		self.t += dt
		x = self.data.pop(0)
		q,r = divmod(x,16)
		if q in [8,9]: # note off or on
			event_type = {8:"off", 9:"on"}[q]
			note = self.data.pop(0)
			velocity = self.data.pop(0)
			return Event(event_type,self.t,note)
		elif hex_chars[q] in ["a","b","e"]:
			self.data=self.data[2:]
		elif hex_chars[q] in ["c","d"]:
			self.data.pop(0)
		elif hex_chars[q] == "f":
			if hex_chars[q] == "f":
				self.data.pop(0)
			l = self.get_varlen()
			self.data = self.data[l:]
		return None

	def read(self):
		events = []
		while self.data:
			e = self.get_event()
			if e:
				events.append(e)
		return Track(events)

		

if __name__ == "__main__":

	import pickle

	names = ["Fugue2","Fugue3"]
	key_shifts = [-3,-1]
	time_incrs = [60,60]
	for name,ks,dt in zip(names,key_shifts,time_incrs):

		f = MidiData("data/midi_inputs/"+name+".mid")
		tracks = [T.read() for T in f.tracks]
		top_voice = Song(tracks[1:2])
		other_voices = Song(tracks[2:])
		top_voice = top_voice.to_data(dt)
		other_voices = other_voices.to_data(dt)
		top_voice = [[v+ks for v in time_step] for time_step in top_voice]
		other_voices = [[v+ks for v in time_step] for time_step in other_voices]
		save_path = "data/pickled/{}TopVoice.p".format(name)
		pickle.dump(top_voice,open(save_path,"wb"))
		save_path = "data/pickled/{}OtherVoices.p".format(name)
		pickle.dump(other_voices,open(save_path,"wb"))


