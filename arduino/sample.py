import pandas as pd
import numpy as np
import argparse
import pickle
import utils.config as config

from sklearn.model_selection import train_test_split
from utils.data_loader import build_vocab, Vocabulary
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def main(args):
	mic = []
	prox = []
	model = pickle.load(open(args.model, 'rb'))
	with open(args.sample) as file:
		vals = file.read()
		vals = vals[:-2]
		vals = vals.split('\n')
	for val in vals:
		d = val.split(',')
		mic.append(int(d[0]))
		prox.append(int(d[1]))

	micWindows = []
	proxWindows = []
	for i in range(len(mic) - config.WINDOW + 1):
		micWindows.append(mic[i:i+config.WINDOW])
		proxWindows.append(mic[i:i+config.WINDOW])

	data = pd.DataFrame({
		'raw_mic':micWindows,
		'raw_prox':proxWindows,
	})
	diff = [1, -1]
	tmp = [np.convolve(diff, x, 'full') for x in proxWindows]
	mic = pd.DataFrame(data['raw_mic'].values.tolist())
	prox = pd.DataFrame(data['raw_prox'].values.tolist())
	conv = pd.DataFrame(tmp)
	X = pd.concat([prox, mic, conv], axis=1)
	print X.head()
	y = model.predict(X)
	print y

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--sample', type=str, required=True)
	parser.add_argument('--model', type=str, required=True)
	args = parser.parse_args()
	main(args)