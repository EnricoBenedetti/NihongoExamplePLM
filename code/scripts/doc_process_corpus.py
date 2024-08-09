from spacy.tokens import DocBin
from datasets import load_dataset
from tqdm import tqdm
import spacy
import ginza
import pandas as pd
import numpy as np 
import itertools

# run
# nohup python doc_process_corpus.py &> log_doc_process.out &


def main():
    # process into multiple docbins of 1M sentences each. ordered.
    #read the dataset
    dataset = load_dataset("bennexx/jp_sentences") # 2 mins to download
    
    nlp = spacy.load("ja_ginza")
    df = dataset['train'].to_pandas()
    # it's not using cuda?
    # process it again
    
    # Serialize vocab
    
    nlp.vocab.to_disk('docs/jp_vocab')

    batch_size = 1000000
    for i, batch_start in enumerate(range(0, len(df), batch_size)):
        if i !=12:
            continue
        
        batch_end = batch_start + batch_size
        docs = nlp.pipe(df['sentence'][batch_start:batch_end], n_process=2)
        print(f'batch is from {batch_start}, {batch_end}')
        # process a batch of docs
        #current_batch = [next(docs) for _ in tqdm(range(batch_start, batch_end))]
        #current_batch = itertools.islice(docs, start, stop[, step])
        # create a docbin
        doc_bin = DocBin(store_user_data=True, docs=tqdm(docs))
        bytes_data = doc_bin.to_bytes()
        
        with open(f'docs/docs{i}_test.bin', 'wb') as file:
            # Write the bytes object to the file
            file.write(bytes_data)
        print(f'saved bin number {i}')

    #reading is in read_doc_bins.ipynb

if __name__== '__main__':
    main()