import json
import numpy as np
import matplotlib.pyplot as plt
import random

datasetFolder = 'dev'
datasetFn = '{}/{}-v2.0.json'.format(datasetFolder, datasetFolder)
analysisFn = '{}/analysis-10.json'.format(datasetFolder)

with open(datasetFn) as f:
	dataset = json.load(f)
with open(analysisFn) as f:
	analysis = json.load(f)

flatten = []
data = dataset['data']
for articleIndex, article in enumerate(data):
	title = article['title']
	paragraphs = article['paragraphs']
	for paragraphIndex, paragraph in enumerate(paragraphs):
		context = paragraph['context']
		qas = paragraph['qas']
		for qaIndex, qa in enumerate(qas):
			question = qa['question']
			id = qa['id']
			answers = qa['answers']
			is_impossible = qa['is_impossible']
			qa['preds'] = analysis[id]
			if is_impossible:
				true_preds = list(filter(lambda x:x['answer']=='', qa['preds']))
			else:
				true_answers = list(map(lambda x:x['text'], answers))
				true_preds = list(filter(lambda x:x['answer'] in true_answers, qa['preds']))
			
			if len(true_preds)==0:
				true_pred = {'prob':-random.random()*0.1 - 0.1}
				true_pred_index = -1
			else:
				true_pred = true_preds[0]
				true_pred_index = qa['preds'].index(true_pred)


			true_pred_prob = true_pred['prob']

			top_pred = qa['preds'][0]
			if true_pred == top_pred:
				continue
				# top_pred = qa['preds'][1]
			
			top_pred_prob = top_pred['prob']

			flatten.append({
				'top_pred_prob': top_pred_prob, 
				'top_pred': top_pred,
				'true_pred_prob': true_pred_prob,
				'true_pred': true_pred,
				'true_pred_index': true_pred_index,
				'title': title,
				'context': context,
				'question': question,
				'is_impossible': is_impossible,
				'answers': answers,
				'preds': qa['preds']
			})


data = map(lambda d:(d['top_pred_prob'], d['true_pred_prob']), flatten)
data = np.array(list(data))
plt.scatter(data[:,0], data[:,1], alpha=0.2)
plt.show()



with open('flatten.js', 'w') as f:
	f.write('let flatten = \n')
	json.dump(flatten, f, indent=2)