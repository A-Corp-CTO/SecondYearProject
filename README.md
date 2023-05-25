# SYP

## Folders in this Repository

```$ 
conll2003 "https://github.com/zliucr/CrossNER/tree/main/ner_data/conll2003"
ai "https://github.com/zliucr/CrossNER/tree/main/ner_data/ai"
politics "https://github.com/zliucr/CrossNER/tree/main/ner_data/politics"
literature "https://github.com/zliucr/CrossNER/tree/main/ner_data/literature"
music "https://github.com/zliucr/CrossNER/tree/main/ner_data/music"
science "https://github.com/zliucr/CrossNER/tree/main/ner_data/science"
```

## Requirements
```$ 
conda                     23.3.1           py39haa95532_0
imbalanced-learn          0.10.1             pyhd8ed1ab_0    conda-forge
ipykernel                 6.19.2           py39hd4e2768_0
ipython                   8.12.0           py39haa95532_0
joblib                    1.1.1            py39haa95532_0
matplotlib                3.7.1            py39haa95532_1
numpy                     1.24.3           py39hf95b240_0
pandas                    1.5.3            py39hf11a4ad_0
pip                       23.0.1           py39haa95532_0
python                    3.9.16               h6244533_2
pytorch                   1.12.1          cpu_py39h5e1f01c_1
regex                     2022.7.9         py39h2bbff1b_0
requests                  2.29.0           py39haa95532_0
scikit-learn              1.2.2            py39hd77b12b_0
scipy                     1.10.1           py39h321e85e_0
tokenizers                0.11.4           py39he5181cf_1
transformers              4.27.1             pyhd8ed1ab_0    conda-forge
yaml                      0.2.5                he774522_0
zipp                      3.11.0           py39haa95532_0
```
## Project Description 



## Preproducing the results
To run an experiment there are 3 files needed. Firstly we need to modify the rnn.py or rnn_random_data.py (only use rnn_random_data for experiments with adding random data). In the rnn.py file we need to specify the files used in the training and a name given to to model that will be saved. Once that is done we need to modify the predictions.py

For that file we need to specify a model that is used to make the predictions and a name given to the file created with the predictions. To run that file we need to use the command line and input this : python predictions.py "Path to file with golden labels".

Lastly when we want to test the results we need to run the span_f1.py file. There is nothing to modify before running it. Simply input into the command line :
python span_f1.py "Path to golden labels" "Path to predictions"
