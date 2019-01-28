import os
import argparse

class Vocabulary(object):
	def __init__(self):
		#TODO: add windowing
		self.micSegment = []
		self.proxSegment = []
		self.ids = {}

	def __len__(self):
		return len(self.windows)

def build_vocab(data_path, win_len):
	vocab = Vocabulary()
	categories = ["notEating", "eating"]
	for cat in categories:
		for filename in os.listdir(data_path + cat):
			mic = []
			prox = []
			with open(data_path + cat + "/" + filename) as file:
				vals = file.read()
				vals = vals[:-2] # remove the new line character at end of file
				vals = vals.split('\r\n')
				vals.pop(0) # remove first line x00
				vals.pop(0) # remove timeout
				for val in vals:
					x, y = val.split(',')
					mic.append(x)
					prox.append(y)
			vocab.micSegment.append(mic)
			vocab.proxSegment.append(prox)
			vocab.ids[len(vocab.micSegment)-1] = cat

def main(args):
	vocab = build_vocab(args.data_path, args.win_len)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str,
		default='../data/', help='path to data files')
	parser.add_argument('--win_len', type=int,
		default=20, help='length of the window to process')
	args = parser.parse_args()
	main(args)