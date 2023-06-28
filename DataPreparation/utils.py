import pandas as pd
import numpy as np


def upload_sparse_cols(df, threshold=0.5):
    '''Uploads sparse columns that were selected on training data, use downloader to apply for prediction or new data.'''
    sparse_cols = df.columns[df.notna().mean()<=threshold]
    with open('sparse_cols_drop.txt','w') as f:
        for col in sparse_cols:
            f.write(f'{col}\n')

def download_sparse_cols():
    with open('sparse_cols_drop.txt','r') as f:
        sparse_cols = [col.replace('\n','') for col in f.readlines()]
    return sparse_cols

def make_aggregator(df):
    numeric_cols = df.columns[df.dtypes!=object]
    object_cols  = df.columns[df.dtypes==object]
    
    aggregations = dict()
    for num_col in numeric_cols:
        aggregations[num_col] = 'sum'

    for text_col in object_cols[1:]:
        aggregations[text_col] = ' '.join
        
    return aggregations