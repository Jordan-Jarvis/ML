# -*- coding: utf-8 -*-
"""
@author: Jordan
This is an implementation and test of a decision tree and k-folds algorithm, made with numpy
and standard libraries. SKlearn datasets used for testing.
"""
from random import randrange
from csv import reader
import math


def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset


def train_test_split(dataset, split=0.50):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def evaluate_algorithm_single(dataset, algorithm, *args):
	train_set,test_set=train_test_split(dataset,0.80)
	scores = list()
	predicted = algorithm(train_set, test_set, *args)
	actual = [row[-1] for row in test_set]
	for row in test_set:
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
def entropy(groups, classes,b_score):

	n_instances = float(sum([len(group) for group in groups]))
	ent = 0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0.0

		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			if p > 0 :
				score=(p*math.log(p,2))
		ent-=(score*(size/n_instances))
	return ent

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 1, None
    for index in range(len(dataset[0])-1):
    	for row in dataset:
    		groups = test_split(index, row[index], dataset)
    		ent = entropy(groups, class_values,b_score)
    		if ent < b_score:
    			b_index, b_value, b_score, b_groups = index, row[index], ent, groups
    	return {'index':b_index, 'value':b_value, 'groups':b_groups}
    
def to_display(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_display(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_display(left), to_display(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_display(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_display(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[ATTRIBUTE[%s] = %.50s]' % ((depth*'\t', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
        
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

filename = 'iris.csv'
dataset = load_csv(filename)
n_folds = 20
max_depth = 10
min_size = 3
train_set,test_set=train_test_split(dataset,0.50)
tree= build_tree(train_set, max_depth, min_size)
print('  ')
print(tree)
print('  ')
print('Attributes ')
print(dataset[0])
print_tree(tree)
scores_1 = evaluate_algorithm_single(dataset,decision_tree,max_depth,min_size)
print('  ')
print('Single Split')
print('Score: %s' % scores_1[0])
print('Accuracy: %.3f%%' % (sum(scores_1)/float(len(scores_1))))
print('  ')
print('Implementing k-cross validation')
scores_2 = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores_2)
print('Average Accuracy: %.3f%%' % (sum(scores_2)/float(len(scores_2))))


