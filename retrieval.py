import bm25retrieval
import wordembedding
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


def remove_punc(data):
    x = pd.Series(data)
    x = data.str.replace('[^\w\s]', '')
    return x


def lowercasing(data):
    x = pd.Series(data)
    return x.str.lower()


def wordembedretrieval():
    pass


def docs_centroid(doc, model):
    docs_vec = []
    for w in doc:
        if w in model.wv.vocab:
            docs_vec.append(model.wv[w])
        else:
            docs_vec.append(np.zeros(model.wv.vector_size))
    return np.mean(docs_vec, axis=0)


def retrieve(df):
    pass


def get_rel_doc(rel_doc_idx, list_idx):
    ret_rel = []
    for doc_id in rel_doc_idx:
        if doc_id in list_idx:
            ret_rel.append(doc_id)
    return ret_rel


def process_list_idx(top_n_doc):
    list_idx = top_n_doc["Indeks data"].to_list()
    tmp = []
    list_idx = [[int(idx) for idx in lidx.split() if idx.isdigit()]
                for lidx in list_idx]
    list_idx = [i[0] for i in list_idx]
    return list_idx


def evaluation(rel_doc_idx, list_idx, n, n_docs, eval_type, rel_doc_idx_loc=None):
    # get metrics
    # # rel_doc_idx = qr.loc[qr['Query_ID'] == pos]
    # list_idx = top_n_doc["Indeks data"].to_list()
    # tmp = []
    # # print(f"list idx before: {list_idx}")
    # list_idx = [[int(idx) for idx in lidx.split() if idx.isdigit()]
    #             for lidx in list_idx]
    # # print(f"list idx: {list_idx}")
    # list_idx = [i[0] for i in list_idx]
    # print(f"list idx now: {list_idx}")
    num = 0
    score = 0

    if rel_doc_idx_loc is None:
        rel_doc_idx_loc = get_rel_doc(rel_doc_idx, list_idx)

    if eval_type == "precision":
        score = (len(rel_doc_idx_loc)/n_docs)*100
    elif eval_type == 'recall':
        score = (len(rel_doc_idx_loc)/n)*100
    return score


if __name__ == "__main__":
    # load data
    data = None
    if os.path.exists("data") and os.path.exists("data/alls.csv"):
        data = pd.read_csv("data/alls.csv")
        data["Content"] = lowercasing(data["Content"])
        data["Content"] = remove_punc(data["Content"])

    if os.path.exists("model"):
        pass
    else:
        # print(data.head())
        wordembedding.build(data["Content"])
    model = wordembedding.load()

    docs_vec = []
    Q = pd.read_csv('data/querys.csv')
    Q["Questions"] = remove_punc(lowercasing(Q["Questions"]))

    pos = 20
    query_vector = docs_centroid(Q["Questions"][pos].split(), model)

    # df = pd.DataFrame()
    # df["Content"] = data["Content"]

    data["docs_vec"] = data.apply(lambda x: docs_centroid(
        x["Content"].split(), model), axis=1)
    data["cos_dis"] = data.apply(lambda x: cosine(
        query_vector, x["docs_vec"]), axis=1)
    print(data.head())
    df = data.sort_values(by=["cos_dis"]).reset_index(drop=True)
    print(df.head())
    print()
    n = 30
    qr = pd.read_csv('data/qrels.csv')

    doc_ids = qr.loc[qr['Query_ID'] == pos+1]

    top_n_doc = df[:n]

    list_idx = process_list_idx(top_n_doc)
    rel_doc = get_rel_doc(doc_ids["Answer_ID"], list_idx)
    n_rel_doc = len(doc_ids)

    score = evaluation(doc_ids, list_idx, n_rel_doc, n, 'recall', rel_doc)
    print(Q["Questions"][0])
    print()
    print(top_n_doc)
    print()
    print(score)
