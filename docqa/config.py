from os.path import join, expanduser, dirname

"""
Global config options
"""

VEC_DIR = './data_raw/glove'
SQUAD_SOURCE_DIR = '.data_raw'
SQUAD_TRAIN = join(SQUAD_SOURCE_DIR, "train-v2.0.json")
SQUAD_DEV = join(SQUAD_SOURCE_DIR, "dev-v2.0.json")


TRIVIA_QA = join("data", "triviaqa")
TRIVIA_QA_UNFILTERED = join("data", "triviaqa-unfiltered")
LM_DIR = join("data", "lm")
DOCUMENT_READER_DB = join("data", "doc-rd", "docs.db")


CORPUS_DIR = 'data'
