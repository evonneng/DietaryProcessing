import pandas as pd
import numpy as np
import argparse
import pickle

from sklearn.model_selection import train_test_split
from utils.data_loader import build_vocab, Vocabulary
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def main(args):
	vocab = build_vocab(args.data_path, args.win_len)
	data = pd.DataFrame({
		'label':vocab.labels,
		'raw_mic':vocab.micWindows,
		'raw_prox':vocab.proxWindows,
	})
	diff = [1, -1]
	tmp = [np.convolve(diff, x, 'full') for x in vocab.proxWindows]

	mic = pd.DataFrame(data['raw_mic'].values.tolist())
	prox = pd.DataFrame(data['raw_prox'].values.tolist())
	conv = pd.DataFrame(tmp)
	X = pd.concat([prox, mic, conv], axis=1)

	y = data['label']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	clf = RandomForestClassifier(n_estimators=20) #TODO should be way higher
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	pickle.dump(clf, open(args.output, 'wb'))
	print("saved to:",args.output)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str,
		default='data/', help='path to data files')
	parser.add_argument('--win_len', type=int,
		default=20, help='length of the window to process')
	parser.add_argument('--output', type=str, required=True,
		help='filename to write model out to')
	args = parser.parse_args()
	main(args)