

from Scripts import span_f1


import matplotlib.pyplot as plt
def learning_curve(golds, preds):
	scores = []
	for gold, pred in zip(golds, preds):
		scores.append(span_f1.getInstanceScores(gold, pred))
	return scores


size = [10, 20, 30, 40, 50, 60]
golds = ['Data/ai/changed_test.txt', 'Data/ai/changed_test.txt', 'Data/ai/changed_test.txt']
preds = ['Predictions/random_ai_20_predictions.txt', 'Predictions/random_ai_50_predictions.txt', 'Predictions/random_ai_60_predictions.txt']

scores = learning_curve(golds, preds)

plt.subplots(1, figsize=(10,10))
plt.plot(size, scores, '--', color="#111111",  label="Learning curve of baseline model")
