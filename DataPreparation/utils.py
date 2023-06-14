import pandas as pd
import numpy as np


def find_sparse_cols(df, threshold=0.5, verbose=True):
    sparse_cols = df.columns[df.notna().mean()>=threshold]
    if verbose:
        dropped_cols = df.columns[df.notna().mean() < threshold]
        print(f'\nTo drop:\n{df[dropped_cols].notna().mean()}')
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