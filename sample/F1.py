"""搜索阈值以获得最佳F1
"""
import numpy as np
from sklearn.metrics import f1_score, roc_curve, accuracy_score

y_label = ([1, 1, 1, 2, 2, 2])
y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])


fpr, tpr, thresholds = roc_curve(y_label, y_pre, pos_label=2)
accuracy_scores = []
for thresh in thresholds:
    accuracy_scores.append(f1_score(y_label,
                                        [1 if m > thresh else 2 for m in y_pre]))

accuracies = np.array(accuracy_scores)
max_accuracy = accuracies.max()
max_accuracy_threshold =  thresholds[accuracies.argmax()]
