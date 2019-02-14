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
	vocab = build_vocab(args.data_path, config.WINDOW)
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
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	pickle.dump(clf, open('models/' + args.output, 'wb'))
	print("saved to:",'models/' + args.output)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str,
		default='data/', help='path to data files')
	parser.add_argument('--output', type=str, required=True,
		help='filename to write model out to')
	args = parser.parse_args()
	main(args)