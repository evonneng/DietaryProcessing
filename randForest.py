import pandas as pd
import argparse

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
	mic = pd.DataFrame(data['raw_mic'].values.tolist())
	prox = pd.DataFrame(data['raw_prox'].values.tolist())
	X = pd.concat([prox,mic], axis=1)
	# print X.head() 

	y = data['label']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	clf = RandomForestClassifier(n_estimators=10) #TODO should be way higher
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str,
		default='data/', help='path to data files')
	parser.add_argument('--win_len', type=int,
		default=5, help='length of the window to process')
	args = parser.parse_args()
	main(args)