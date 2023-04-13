import sys
from rnn import read_data, data2feats
import pickle
import torch
import numpy as np

PAD = "PAD"
id_to_token = [PAD]
token_to_id = {PAD: 0}
id_to_tag = [PAD]
tag_to_id = {PAD: 0}

def load_model():
    with open('Models/baseline_model.sav', 'rb') as f:
        pipeline = pickle.load(f)
        return pipeline

model = load_model()

def run_eval(feats_batches, labels_batches):
    model.eval()
    predictions = []
    for sents, labels in zip(feats_batches, labels_batches):
        output_scores = model.forward(sents)
        predicted_tags  = torch.argmax(output_scores, 2)
        predictions.append(predicted_tags.tolist())
    return predictions

for devPath in sys.argv[1:]:
    BATCH_SIZE=1
    dev_data=read_data(devPath)
    for tokens, tags in dev_data:
        for token in tokens:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
                id_to_token.append(token)
        for tag in tags:
            if tag not in tag_to_id:
                tag_to_id[tag] = len(tag_to_id)
                id_to_tag.append(tag)

    NWORDS = len(token_to_id)
    NTAGS = len(tag_to_id)
    max_len=max([len(x[0]) for x in dev_data])
    dev_feats, dev_labels = data2feats(dev_data, token_to_id, tag_to_id)
    num_batches2 = int(len(dev_feats)/BATCH_SIZE)

    dev_feats_batches = dev_feats[:BATCH_SIZE*num_batches2].view(num_batches2, BATCH_SIZE, max_len)
    dev_labels_batches = dev_labels[:BATCH_SIZE*num_batches2].view(num_batches2, BATCH_SIZE, max_len)
    pred_tags = run_eval(dev_feats_batches, dev_labels_batches)

final_pred = pred_tags[0]
for i in pred_tags[1:]:
    final_pred.extend(i)

save = True
if save == True:
	with open("literature_predicted_tags",'w') as file:
	    for feats, labels in zip(dev_feats, final_pred):
	        #print(len(feats), len(labels))
	        for feat, label in zip(feats, labels):
	            #print(len(feat), len(label))
	            feat_key, feat_val = list(token_to_id.keys()), list(token_to_id.values())
	            label_key, label_val = list(tag_to_id.keys()), list(tag_to_id.values())
	            id_feat, id_label = feat_val.index(feat), label_val.index(label)
	            if id_feat != 0:
	            	file.write(feat_key[id_feat] + '\t' + label_key[id_label] + '\n')
	        file.write('\n')