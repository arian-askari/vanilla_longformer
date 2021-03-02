# Longformer-based summarization for case law retrieval
The implementation is based on *CEDR: Contextualized Embeddings for Document Ranking*, and we added some modifications for be able to use longformer in the pairwise setting for ranking. We will update the implementation and move the 'train_longformer.py' file to 'train.py'. Most of this readme is copied from CEDR github and will be updated before publishing the paper.
## Getting started

This code is tested on Python 3.6. Install dependencies using the following command:

```
pip install -r requirements.txt
```

You will need to prepare files for training and evaluation. Many of these files are available in
`data/wt` (TREC WebTrack) and `data/robust` (TREC Robust 2004).

**qrels**: a standard TREC-style query relevant file. Used for identifying relevant items for
training pair generation and for validation (`data/wt/qrels`, `data/robust/qrels`).

**train_pairs**: a tab-deliminted file containing pairs used for training. The training process
will only use query-document pairs found in this file. Samples are in `data/{wt,robust}/*.pairs`.
File format:

```
[query_id]	[doc_id]
```

**valid_run**: a standard TREC-style run file for re-ranking items for validation. The `.run` files used for re-ranking are available in `data/{wt,robust}/*.run`. Note that these runs are using the default parameters, so they do not match the tuned results shown in Table 1.

**datafiles**: Files containing the text of queries and documents needed for training, validation,
or testing. Should be in tab-delimited format as follows, where `[type]` is either `query` or `doc`,
`[id]` is the identifer of the query or document (e.g., `132`, `clueweb12-0206wb-59-32292`), and
`[text]` is the textual content of the query or document (no tabs or newline characters,
tokenization done by `BertTokenizer`).

```
[type]  [id]  [text]
```

Queries for WebTrack and Robust are available in `data/wt/queries.tsv` and `data/robust/queries.tsv`.
Document text can be extracted from an index using `extract_docs_from_index.py` (be sure to use an
index that has appropriate pre-processing). The script supports both Indri and Lucene (via Anserini)
indices. See instructions below for help installing pyndri or Anserini.

Examples:

```
# Indri index
awk '{print $3}' data/robust/*.run | python extract_docs_from_index.py indri PATH_TO_INDRI_INDEX > data/robust/documents.tsv
# Lucene index (should be built with Anserini and the -storeTransformedDocs)
awk '{print $3}' data/robust/*.run | python extract_docs_from_index.py lucene PATH_TO_LUCENE_INDEX > data/robust/documents.tsv
```

## Running Vanilla BERT

To train a Vanilla BERT model, use the following command:

```
python train.py \
  --model vanilla_bert \
  --datafiles data/queries.tsv data/documents.tsv \
  --qrels data/qrels \
  --train_pairs data/train_pairs \
  --valid_run data/valid_run \
  --model_out_dir models/vbert
```

## Running Vanilla Longformer

To train a Vanilla Longformer model, use the following command:

```
python train_longformer.py \
  --model vanilla_longformer \
  --datafile_query data/queries.tsv \ 
  --datafile_document data/documents.tsv \
  --qrels data/qrels \
  --train_pairs data/train_pairs \
  --valid_run data/valid_run \
  --model_out_dir models/vlongformer
```
