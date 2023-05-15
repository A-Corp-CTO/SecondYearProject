from rnn_random_data import read_data, data2feats, token_to_id, tag_to_id, max_len, id_to_tag
import pickle
import torch
import sys
#import rnn_random_data
def load_model():
    with open('../Models/random_ai_50.sav', 'rb') as f:
        pipeline = pickle.load(f)
        return pipeline

model = load_model()

def run_eval(feats_batches, labels_batches):
    model.eval()
    match = 0
    total = 0
    for sents, labels in zip(feats_batches, labels_batches):
        output_scores = model.forward(sents)
        predicted_tags  = torch.argmax(output_scores, 2)
        for goldSent, predSent in zip(labels, predicted_tags):
            for goldLabel, predLabel in zip(goldSent, predSent):
                if goldLabel.item() != 0:
                    total += 1
                    if goldLabel.item() == predLabel.item():
                        match+= 1
    return(match/total)

for devPath in sys.argv[1:]:
    BATCH_SIZE=1
    dev_data=read_data(devPath)
    dev_feats, dev_labels = data2feats(dev_data, token_to_id, tag_to_id)
    num_batches2 = int(len(dev_feats)/BATCH_SIZE)
    max_len2=max([len(x[0]) for x in dev_data])

    dev_feats_batches = dev_feats[:BATCH_SIZE*num_batches2].view(num_batches2, BATCH_SIZE, max_len)
    dev_labels_batches = dev_labels[:BATCH_SIZE*num_batches2].view(num_batches2, BATCH_SIZE, max_len)
    pred_tags = run_eval(dev_feats_batches, dev_labels_batches)

save = True
if save == True:
    with open("../Predictions/random_ai_50_predictions.txt",'w',encoding='utf-8') as outfile:
        for batchIdx in range(0, num_batches2):
            input = dev_feats_batches[batchIdx]
            output_scores = model.forward(input)
            output_scores = output_scores.view(BATCH_SIZE * max_len, -1)
            predicted_tags  = torch.argmax(output_scores, 1)
            predicted_tags = predicted_tags.view(BATCH_SIZE, max_len,-1)
            
            for word, pred in zip(dev_data[batchIdx][0],predicted_tags.view(BATCH_SIZE * max_len, -1)):
                if word != 0:
                    outfile.write(word+"\t"+id_to_tag[pred]+"\n")

            outfile.write("\n")