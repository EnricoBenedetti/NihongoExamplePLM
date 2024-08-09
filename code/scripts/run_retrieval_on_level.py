import retrieval
from retrieval import *
import argparse
from utils import get_logger

# nohup python run_retrieval_on_level --target_level 4 --device 4 &

def main():

    parser = argparse.ArgumentParser(description='input the level and the gpu device')
    parser.add_argument('--target_level', type=str, help='N1-N5 target level')
    parser.add_argument('--device', type=int, help='device for models')
    
    args = parser.parse_args()

    log = get_logger()

    config = {'dataset' : "bennexx/jp_sentences",
          'spacy_model': 'ja_ginza',
          'index_file': '/data/enrico_benedetti/nihongoexample/data/corpus_all/jp_sentences/inverted_index.pkl',
          'sense_model': 'bennexx/mirrorwic-cl-tohoku-bert-base-japanese-v3',
          #'diff_model': "/data/enrico_benedetti/nihongoexample/code/models/checkpoint-66"
          'diff_model': 'bennexx/cl-tohoku-bert-base-japanese-v3-jlpt-classifier',
          # 'device': 'cpu',
          'device': args.device,
          'save_dir' : '../../evaluation/outputs/retrieval'
          }

    retr = Retrieval(config)

    target_df = pd.read_csv('../../data/targets/target_words.csv')
    level = args.target_level
    for id, row in tqdm(target_df.iterrows(), desc=f"Processing target words at {level} "):
        log.info(f'level {level}, searching for {row["target_word"]}...')
        target_word = row['target_word']
        context_sentence = row['context_sentence']
        try:
            retr.get_sentence_list(target_word, context_sentence, k=5, target_level=level)
        except Exception as e:
            log.error(e)
    
    del retr


if __name__ == '__main__':
    main()