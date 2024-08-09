# %%
import bz2
import tarfile
import pandas as pd

def main():
    df = None

    jp_sentences_file = './jpn_sentences.tsv.bz2'
    attribution_file = './sentences_base.tar.bz2'
    attribution_file_decompressed = '.' + attribution_file.split('.')[-3] + '.csv'

    # read jp sentences
    with bz2.open(jp_sentences_file, 'rt') as file:
        df = pd.read_csv(file, delimiter='\t', header=None, names=['tatoeba_id','language','sentence'])
    # read attribution (translation or original)
    with tarfile.open(attribution_file, 'r:bz2') as tar:
        tar.extractall()
    df_attribution = pd.read_csv(attribution_file_decompressed, delimiter='\t', header=None, names=['tatoeba_id', 'base_field'])

    # we don't really care about the sentence link so we delete the numbers > 0 (we keep 0, non0, and unknown)

    def process_base_field(field):
        if field == '0':
            return 'original'
        elif field == '\\N':
            return 'unknown'
        else: return 'translation'

    df_attribution['base_field'] = df_attribution['base_field'].map(process_base_field)

    # we only care about the jp sentences
    df_attribution_jp = df_attribution[df_attribution['tatoeba_id'].isin(df['tatoeba_id'])]

    # merge the information
    df = df.merge(df_attribution_jp, on='tatoeba_id')
    # add the source link
    df['source'] = 'https://tatoeba.org/en/downloads'

    # info_columns = ['source', 'tatoeba_id', 'base_field']
    # df['info'] = df[info_columns].to_dict(orient='records')

    # drop language and other info
    # df.drop('language', axis='columns', inplace=True)
    # df_clean = df.drop(info_columns, axis='columns')
    # df_sentence_only = df_clean['sentence']

    # df_clean.to_csv('tatoeba_info.csv')
    # df_sentence_only.to_csv('tatoeba.csv')
    df.to_csv('tatoeba.csv')

if __name__=='__main__':
    main()