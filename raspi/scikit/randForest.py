import pandas as pd
import numpy as np
import argparse
import pickle
import utils.config as config

from sklearn.model_selection import train_test_split
from utils.data_loader import build_vocab, Vocabulary
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def main(args):
	vocab = build_vocab(args.data_path)
	data = pd.DataFrame({
		'label':vocab.labels,
		'lprox':vocab.lprox,
		'rprox':vocab.rprox,
		'x':vocab.x,
		'y':vocab.y,
		'z':vocab.z,
	})
	y = data['label']
	lprox = pd.DataFrame(data['lprox'].values.tolist())
	rprox = pd.DataFrame(data['rprox'].values.tolist())
	xax = pd.DataFrame(data['x'].values.tolist())
	yax = pd.DataFrame(data['y'].values.tolist())
	zax = pd.DataFrame(data['z'].values.tolist())

	X = pd.concat([lprox, rprox, xax, yax, zax], axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	param_grid = {
		'bootstrap': [True],
		'max_depth': [80, 90, 100, 110],
		'max_features': [2, 3],
		'min_samples_leaf': [3, 4, 5],
		'min_samples_split': [8, 10, 12],
		'n_estimators': [100, 200, 300, 1000]
	}
	clf = RandomForestClassifier(n_estimators=100)
	grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
	grid_search.fit(X_train, y_train)
	grid_search.best_params_
	best_grid = grid_search.best_estimator_
	# grid_accuracy = evaluate(best_grid, X_test, y_test)

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