# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: demo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-04-25 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

H_raw = loadmat('./data/h_full_collt_qam.mat')['h_full_collt']
y = loadmat('./data/s_n_o_qam.mat')['s_n_o']

H_module = abs(H_raw)[:, 0, :]

y = y[0]


ratio = 0.8
train_samples = int(len(H_module) * ratio)
test_samples = len(H_module) - train_samples

H_train, H_test = H_module[:train_samples], H_module[-test_samples:]
y_train, y_test = y[:train_samples], y[-test_samples:]

clf = LogisticRegression()
clf.fit(H_train, y_train)

pred_test_lr = clf.predict(H_test)
pred_all_lr = clf.predict(H_module)

print('LogisticRegression Accuracy: {:.4f}'.format(metrics.accuracy_score(y_test, pred_test_lr)))
print('LogisticRegression Confusion matrix:\n', metrics.confusion_matrix(y_test, pred_test_lr))


clf = SVC(kernel='linear', gamma='scale')
clf.fit(H_train, y_train)
pred_test_svm = clf.predict(H_test)
pred_all_svm = clf.predict(H_module)
print('SVM Accuracy: {:.4f}'.format(metrics.accuracy_score(y_test, pred_test_svm)))
print('SVM Confusion matrix:\n', metrics.confusion_matrix(y_test, pred_test_svm))


clf = RandomForestClassifier(n_estimators=200)
clf.fit(H_train, y_train)
pred_test_rf = clf.predict(H_test)
pred_all_rf = clf.predict(H_module)
print('RF Accuracy: {:.4f}'.format(metrics.accuracy_score(y_test, pred_test_rf)))
print('RF Confusion matrix:\n', metrics.confusion_matrix(y_test, pred_test_rf))

clf = MLPClassifier(hidden_layer_sizes=(16, 32, 8,), verbose=False, max_iter=1000, n_iter_no_change=30)
clf.fit(H_train, y_train)
pred_test_mlp = clf.predict(H_test)
pred_all_mlp = clf.predict(H_module)
print('MLP Accuracy: {:.4f}'.format(metrics.accuracy_score(y_test, pred_test_mlp)))
print('MLP Confusion matrix:\n', metrics.confusion_matrix(y_test, pred_test_mlp))

final_predict = {'LR': pred_all_lr, 'SVM': pred_all_svm, 'RF': pred_all_rf, 'NN': pred_all_mlp, 'Truth': y}
savemat('./data/results.mat', final_predict)


out_file = loadmat('./data/results.mat')
print(out_file.keys())
