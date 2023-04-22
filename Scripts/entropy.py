from rnn import read_data, data2feats, token_to_id, tag_to_id, max_len
import pickle
import torch
import sys
import numpy as np

def load_model():
    with open('../Models/baseline_model_ai.sav', 'rb') as f:
        pipeline = pickle.load(f)
        return pipeline

model = load_model()


def prob(data):
    probabilities = []
    model.eval()
    for sentence in data:
        sen = []
        output_scores = model.forward(sentence)
        predicted_tags = (output_scores.softmax(1)).tolist()
        for word in predicted_tags:
            label_probs = {}
            for index, score in enumerate(word):
                key = next((k for k, v in tag_to_id.items() if v == index), None)
                label_probs[key] = score
            sen.append(label_probs)
        probabilities.append(sen)
    return probabilities


def entropy(probabilities):
    scores = []
    for sentence in probabilities:
        sen = []
        for word in sentence:
            entropy = 0.0
            for k, p in word.items():
                if p != 0:
                    entropy -= p * np.log2(p)
            sen.append(entropy.round(2))
        scores.append(sen)
    return scores


for devPath in sys.argv[1:]:
    dev_data = read_data(devPath)
    sentence, sen_labels = [i[0] for i in dev_data], [i[1] for i in dev_data]
    length = [len(i) for i in sentence]
    dev_feats, dev_labels = data2feats(dev_data, token_to_id, tag_to_id)
    probabilities = prob(dev_feats)
    prob2 = []
    for prob, leng in zip(probabilities, length):
        prob2.append(prob[:leng])
    scores = entropy(prob2)


save = False
if save == True:
    with open("../ai_entropy_score4.txt",'w') as outfile:
        threshold = 2.5
        for i, j, k in zip(sentence, sen_labels, scores):
            for word, label, score in zip(i,j,k):
                if score < threshold:
                    outfile.write(f"{word}\t{label}\t{score}\n")
                else:
                    outfile.write(f"{word}\t{label}\t{score}\t{'SNEHA'}\n")
            outfile.write("\n")
