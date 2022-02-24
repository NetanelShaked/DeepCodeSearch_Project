from transformers import AutoTokenizer, AutoModel
import torch
import json
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def convertTabularRecToImg(code, nl, label):    
    d = {}
    for j in range(2):
        for i in range(768):
            d['f'+str(j)+'_'+str(i)] = []

    d['label'] = []

    nl_tokens = [tokenizer.cls_token] + tokenizer.tokenize(nl) + [tokenizer.sep_token]
    code_tokens = [tokenizer.cls_token] + tokenizer.tokenize(code) + [tokenizer.sep_token]

    tokens_ids_nl = tokenizer.convert_tokens_to_ids(nl_tokens)
    tokens_ids_code = tokenizer.convert_tokens_to_ids(code_tokens)
    try:
        context_embeddings_nl = model(torch.tensor(tokens_ids_nl)[None, :])[0]
        context_embeddings_code = model(torch.tensor(tokens_ids_code)[None, :])[0]
    except:
        print('error')
        continue


    for index, embedding in enumerate([context_embeddings_nl, context_embeddings_code]):
        for i, feature in enumerate(embedding[0][0]):
            d['f'+str(index)+'_'+str(i)].append(float(feature))

    d['label'].append(label)
    df = pd.DataFrame(d)
    
    return df
