import json, sys

# pred,
# naprob,

datasetFolder = 'dev'
datasetFn = '{}/{}-v2.0.json'.format(datasetFolder, datasetFolder)
predFn = '{}/pred_alwaysAnswer.json'.format(datasetFolder)
probFn = '{}/na_prob.json'.format(datasetFolder)
analysisFn = '{}/analysis-10.json'.format(datasetFolder)

with open(datasetFn) as f:
	dataset = json.load(f)
with open(predFn) as f:
	preds = json.load(f)
with open(probFn) as f:
	probs = json.load(f)
with open(analysisFn) as f:
	analysis = json.load(f)

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

			pred = preds[id]
			prob = probs[id]
			qa['preds'] = analysis[id]

with open('dataset.js', 'w') as f:
	f.write('let dataset = \n')
	json.dump(dataset, f, indent=2)


			# print(title)
			# print(context)
			# print('-'*30)
			# print('(question id)\t', id)
			# print('(question)\t', question)
			# print('(is impossible)\t', is_impossible)
			# for i,t in enumerate(answers):
			# 	print('(truth)\t\t', t)
			# print('-'*30)
			# print('(pred)\t\t', pred)
			# print('(prob)\t\t', prob)
			# print('-'*60)
