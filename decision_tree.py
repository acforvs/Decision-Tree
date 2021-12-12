import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=None):
    R = len(feature_vector)
    stacked = np.vstack((feature_vector, target_vector))
    stacked = stacked[:, stacked[0].argsort()]

    ones = np.cumsum(stacked[1, :])
    fv = stacked[0, :].copy()
    uniq = np.unique(fv)

    c = Counter(fv)
    counter = c.items()

    cnt = list(map(lambda x: x[1], sorted(counter, key=lambda x: x[0])))
    occ = np.cumsum(cnt)

    left_ind, right_ind = 0, len(occ)
    if min_samples_leaf is not None:
        left_ind = np.searchsorted(occ, min_samples_leaf)
        right_ind = np.searchsorted(
            occ, occ[-1] - min_samples_leaf, side='right')

    def gini(i):
        R_l, R_r = occ[i], R - occ[i]
        p1_l = ones[occ[i] - 1] / R_l
        p0_l = 1 - p1_l
        p1_r = (ones[-1] - ones[occ[i] - 1]) / R_r
        p0_r = 1 - p1_r

        return -R_l / R * (1 - p1_l**2 - p0_l**2) - R_r / \
            R * (1 - p1_r**2 - p0_r**2)

    r = list(range(left_ind, right_ind))
    if not r:
        return np.array([]), np.array([]), None, None
    ginis = np.vectorize(gini)(r)[:-1]
    if not ginis.shape or not ginis.shape[0]:
        return np.array([]), np.array([]), None, None
    thresholds = ((uniq + np.roll(uniq, -1)) / 2)[:-1]

    return thresholds, ginis, thresholds[ginis.argmax()], ginis.max()


class DecisionTree():
    def __init__(
            self,
            feature_types,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None):
        if np.any(
                list(map(lambda x: x != 'real' and x != 'categorical', feature_types))):
            raise ValueError('There is unknown feature type')

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _fit_node(self, sub_X, sub_y, node):
        if 'depth' not in node.keys():
            node['depth'] = 0

        if np.all(sub_y == sub_y[0]):
            node['type'] = 'terminal'
            node['class'] = sub_y[0]
            return

        if self._max_depth is not None and node['depth'] == self._max_depth:
            node['type'] = 'terminal'
            node['class'] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and sub_X.shape[1] < self._min_samples_split:
            node['type'] = 'terminal'
            node['class'] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == 'real':
                feature_vector = sub_X[:, feature]
            elif feature_type == 'categorical':
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                        ratio[key] = current_click / current_count
                    else:
                        ratio[key] = 0

                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(
                    zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(
                    list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(
                feature_vector, sub_y, self._min_samples_leaf)
            if threshold and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if feature_type == 'real':
                    threshold_best = threshold
                elif feature_type == 'categorical':
                    threshold_best = list(map(lambda x: x[0], filter(
                        lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node['type'] = 'terminal'
            node['class'] = Counter(sub_y).most_common(1)[0][0]
            return

        node['type'] = 'nonterminal'

        node['feature_split'] = feature_best
        if self._feature_types[feature_best] == 'real':
            node['threshold'] = threshold_best
        elif self._feature_types[feature_best] == 'categorical':
            node['categories_split'] = threshold_best
        else:
            raise ValueError
        node['left_child'], node['right_child'] = {
            'depth': node['depth'] + 1}, {'depth': node['depth'] + 1}
        self._fit_node(sub_X[split], sub_y[split], node['left_child'])
        self._fit_node(sub_X[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node['right_child'])

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        if node['type'] == 'nonterminal':
            feature = node['feature_split']
            if self._feature_types[feature] == 'real':
                if x[feature] < node['threshold']:
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
            elif self._feature_types[feature] == 'categorical':
                if x[feature] in node['categories_split']:
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
            else:
                raise ValueError
        else:
            raise ValueError

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        X = np.array(X)
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
