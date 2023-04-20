from rnn import read_data, data2feats
import pickle
import torch
import sys
import rnn
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
                if score > 0:
                    ff = index
                    key = next((k for k, v in rnn.tag_to_id.items() if v == ff), None)
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
    dev_feats, dev_labels = data2feats(dev_data, rnn.token_to_id, rnn.tag_to_id)
    probabilities = prob(dev_feats)
    print(entropy(probabilities))


# take dev feats as column1 -> dev_feats
# predictions as column 2 -> 
# if entropy above threshold column 3

# with open('ai_entropy.txt', 'w') as file:
