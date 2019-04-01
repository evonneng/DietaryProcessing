import os
import argparse
import config

class Vocabulary(object):
	def __init__(self):
		self.lprox = []
		self.rprox = []
		self.x = []
		self.y = []
		self.z = []
		self.labels = {}

	def __len__(self):
		return len(self.labels)

def build_vocab(data_path):
	win_len = config.WINDOW
	vocab = Vocabulary()
	categories = ["notEating", "eating"]
	index = 0
	for cat in categories:
		for filename in os.listdir(data_path + cat):
			l = []
			r = []
			x = []
			y = []
			z = []
			with open(data_path + cat + "/" + filename) as file:
				print 'processing:', data_path + cat + '/' + filename
				vals = file.read()
				vals = vals.replace("[", "")
				vals = vals.split("]")[:-1]
			for v in vals:
				v = v.split(", ")
				l.append(float(v[0]))
				r.append(float(v[1]))
				x.append(float(v[2]))
				y.append(float(v[3]))
				z.append(float(v[4]))

			for i in range(len(vals) - win_len):
				vocab.lprox.append(l[i:i+20])
				vocab.rprox.append(r[i:i+20])
				vocab.x.append(r[i:i+20])
				vocab.y.append(y[i:i+20])
				vocab.z.append(z[i:i+20])
				vocab.labels[index] = cat
				index += 1

	return vocab


def main(args):
	vocab = build_vocab(args.data_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str,
		default='../data/', help='path to data files')
	args = parser.parse_args()
	main(args)
