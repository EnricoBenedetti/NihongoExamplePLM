# %%
import utils
from utils import *
import datasets
from transformers import pipeline
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
from tqdm import tqdm

def main():
    # %%
    # load the dataset
    corpus = load_dataset("bennexx/jp_sentences", split="train")
    # load the difficulty classifier
    # Use a pipeline as a high-level helper
    pipe = pipeline("text-classification", model="bennexx/cl-tohoku-bert-base-japanese-v3-jlpt-classifier", device = 7)
    # make the sentences into a Dataset

    # use the pipeline to get the labels

    # save the file as corpus_level.csv

    # %%
    # KeyDataset is a util that will just output the item we're interested in.
    labels = []
    for i, out in tqdm(enumerate(pipe(KeyDataset(corpus, "sentence"), batch_size=1000))):
        labels.append(out['label'])
        del out
    corpus = corpus.add_column('level', labels)

    # %%
    df = pd.DataFrame(corpus)
    df.to_csv("../../data/corpus_all/jp_sentences/corpus_level.csv", index=False)

if __name__=="__main__":
    main()
