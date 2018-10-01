# python -m nltk.downloader punkt stopwords

# mkdir -p ./data
# mkdir -p ./data/glove
# cd ./data/glove
# wget http://nlp.stanford.edu/data/glove.840B.300d.zip
# unzip glove.840B.300d.zip

# mkdir -p ./data_raw/squad
# cd ./data_raw/squad
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json




export PYTHONPATH=${PYTHONPATH}:`pwd`
export CUDA_VISIBLE_DEVICES=0

# mkdir -p ./data_raw/squad
# cd ./data_raw/squad
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# mkdir -p ./data_raw
# mkdir -p ./data_raw/glove
# cd ./data_raw/glove
# wget http://nlp.stanford.edu/data/glove.840B.300d.zip
# unzip glove.840B.300d.zip

DATARAW='data_raw/squad/'
PREDFILE='./pred.json'


# python docqa/squad/build_squad_dataset.py --train_file train-v2.0.json --dev_file dev-v2.0.json
python docqa/squad/build_squad_dataset.py --basedir $DATARAW

######## TRAIN ########
python docqa/scripts/ablate_squad.py confidence --num-epochs 25 --no-tfidf model


# #(OLD, NOT-USED) EVALUATE v1.1
# python docqa/eval/squad_eval.py -o $PREDFILE -c dev model*
# python eval_squad.py $DATARAW'dev-v2.0.json' $PREDFILE > eval.json


######## EVALUATE v2.0 ########
# #inspired by commands in https://worksheets.codalab.org/bundles/0x84e5b97e6ccb4c27ae698d82344a787b/

# #generate prediction, alwaysAnswer and probability file.
# python docqa/run/run_json.py model* $DATARAW'dev-v2.0.json' pred.json --na-prob-file na_prob.json --always-answer-file pred_alwaysAnswer.json
# #evaluate on pred.json
# python evaluate-v2.0.py $DATARAW'dev-v2.0.json' pred.json -o eval.json
# #evaluate on pred_alwaysAnswer.json and na_prob.json, with varying threshold
# python evaluate-v2.0.py $DATARAW'dev-v2.0.json' pred_alwaysAnswer.json -o eval_pr.json -n na_prob.json -p plots

# # evaluate the same, but output to stdout/stderr
python docqa/run/run_json.py model* $DATARAW'dev-v2.0.json' pred.json --na-prob-file na_prob.json --always-answer-file pred_alwaysAnswer.json
python evaluate-v2.0.py $DATARAW'dev-v2.0.json' pred.json
python evaluate-v2.0.py $DATARAW'dev-v2.0.json' pred_alwaysAnswer.json -n na_prob.json

# # test run_json.py
python docqa/run/run_json.py \
model* \
$DATARAW'dev-v2.0.json' \
/dev/null \
--na-prob-file /dev/null \
--always-answer-file /dev/null \
--analysis-file ./analysis.json





python evaluate-v2.0.py \
$DATARAW'dev-v2.0.json' pred_alwaysAnswer.json \
--na-prob-file na_prob.json \
--na-prob-thresh 0.27222198247909546 \
--out-image-dir plot \
--verbose
