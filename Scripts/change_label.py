import codecs


ai = {'I-location', 'I-misc', 'B-university', 'I-algorithm', 'I-field', 'B-metrics', 'I-product', 'I-researcher', 'B-misc', 'I-programlang',
	 'B-task', 'I-metrics', 'I-organisation', 'B-researcher', 'O', 'I-person', 'B-product', 'B-country', 'B-field', 'I-task', 'B-person',
	 'I-university', 'B-programlang', 'I-conference', 'B-organisation', 'I-country', 'B-location', 'B-conference', 'B-algorithm'}


conll = {'B-misc', 'I-location', 'I-misc', 'B-person', 'B-organisation', 'B-location', 'I-organisation', 'O', 'I-person'}


change_ai = {'I-location':'I-location','I-misc':'I-misc','B-university': 'B-organisation', 'I-algorithm': 'I-misc', 'I-field': 'I-misc',
             'B-metrics': 'B-misc', 'I-product':'I-misc', 'I-researcher':'I-person', 'B-misc':'B-misc', 'I-programlang':'I-misc','B-task':'B-misc',
             'I-metrics':'I-misc', 'I-organisation':'I-organisation', 'B-researcher':'B-person', 'O':'O', 'I-person':'I-person', 'B-product':'B-misc',
             'B-country':'B-location', 'B-field':'B-misc', 'I-task':'I-misc', 'B-person':'B-person','I-university':'I-location', 'B-programlang':'B-misc',
             'I-conference':'I-organisation', 'B-organisation':'B-organisation', 'I-country':'I-location', 'B-location':'B-location', 'B-conference':'B-organisation',
             'B-algorithm':'B-misc'}

change_labels = {'I-location': 'I-location', 'I-misc': 'I-misc', 'B-university': 'B-organisation', 'I-algorithm': 'I-misc', 'I-field': 'I-misc',
                 'B-metrics': 'B-misc', 'I-product': 'I-misc', 'I-researcher': 'I-person', 'B-misc': 'B-misc', 'I-programlang': 'I-misc',
                 'B-task': 'B-misc', 'I-metrics': 'I-misc', 'I-organisation': 'I-organisation', 'B-researcher': 'B-person', 'O': 'O', 'I-person': 'I-person',
                 'B-product': 'B-misc', 'B-country': 'B-location', 'B-field': 'B-misc', 'I-task': 'I-misc', 'B-person': 'B-person', 'I-university': 'I-organisation',
                 'B-programlang': 'B-misc', 'I-conference': 'I-organisation', 'B-organisation': 'B-organisation', 'I-country': 'I-location', 'B-location': 'B-location',
                 'B-conference': 'B-organisation', 'B-algorithm': 'B-misc', 'B-event': 'B-misc', 'B-musicgenre': 'B-misc', 'I-musicalartist': 'I-person',
                 'I-discipline': 'I-misc', 'I-writer': 'I-person', 'I-election': 'I-misc', 'I-album': 'I-misc', 'B-politician': 'B-person', 'B-theory': 'B-misc',
                 'B-band': 'B-organisation', 'B-chemicalcompound': 'B-misc', 'I-politicalparty': 'I-organisation', 'I-musicgenre': 'I-misc',
                 'I-astronomicalobject': 'I-misc', 'B-album': 'B-misc', 'I-scientist': 'I-person', 'B-poem': 'B-misc', 'I-enzyme': 'I-misc', 'I-theory': 'I-misc',
                 'B-literarygenre': 'B-misc', 'B-scientist': 'B-person', 'I-magazine': 'I-misc', 'I-song': 'I-misc', 'I-award': 'I-misc', 'B-chemicalelement': 'B-misc',
                 'I-poem': 'I-misc', 'B-protein': 'B-misc', 'I-event': 'I-misc', 'I-protein': 'I-misc', 'B-song': 'B-misc', 'B-academicjournal': 'B-misc',
                 'B-writer': 'B-person', 'B-award': 'B-misc', 'B-magazine': 'B-misc', 'I-chemicalcompound': 'I-misc', 'I-chemicalelement': 'I-misc','I-book': 'I-misc', 'B-enzyme': 'B-misc',
                 'I-band': 'I-organisation', 'B-discipline': 'B-misc', 'I-literarygenre': 'I-misc', 'B-book': 'B-misc', 'B-astronomicalobject': 'B-misc',
                 'I-academicjournal': 'I-misc', 'B-election': 'B-misc', 'B-musicalartist': 'B-person', 'I-politician': 'I-person', 'B-politicalparty': 'B-organisation',
                 'B-musicalinstrument': 'B-misc', 'I-musicalinstrument': 'I-misc'}




def change_label(file_name):
   
    with open('../Data/music/changed_train.txt', 'w') as outfile:
        for line in codecs.open(file_name, encoding='utf-8'):
            line = line.strip()

            if line:
                tok = line.split('\t')
                word =tok[0]
                tag = tok[1]

                changed_tag = change_labels[tag]

                outfile.write(word+"\t"+changed_tag+"\n")
            else:
                outfile.write("\n")
change_label('../Data/music/train.txt')