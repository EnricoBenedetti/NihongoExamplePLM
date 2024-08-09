## this class will load index etc, and provide results...
import utils
import torch
from inverted_index import InvertedIndex
from datasets import load_dataset, Dataset
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import \
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline
import os

class Retrieval():
    def __init__(self, config):
        # assign resources
        self.config = config
        
        self.nlp = spacy.load(config['spacy_model'])

        # if scoring ai things, no dataset or index needed
        try:
            if config['generation'] is True:
                print('not using index or dataset')
        except KeyError:
            self.dataset = load_dataset(config['dataset'])
            self.df = self.dataset['train'].to_pandas()
            self.index = InvertedIndex(nlp=self.nlp)
            self.index.load_inverted_index(filename=config['index_file'])


        self.save_dir = config['save_dir']
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.makedirs(self.save_dir + '/debug')
        
        # if self.nlp is None:
        #     self.nlp = nlp
        #     self.df = df
        #     self.index = index

        # load sense model
        self.sense_model = AutoModel.from_pretrained(config['sense_model'])
        self.sense_tokenizer = AutoTokenizer.from_pretrained(config['sense_model'])
        
        self.sense_model.eval()
        # load difficulty classifier
        self.id2label = {0: "N5", 1: "N4", 2: "N3", 3: "N2", 4: "N1"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.diff_model = AutoModelForSequenceClassification.from_pretrained(config['diff_model'], num_labels=5, id2label=self.id2label, label2id=self.label2id)
        self.diff_tokenizer = AutoTokenizer.from_pretrained(config['diff_model'])
        
        self.diff_model.eval()

        self.device = torch.device(config['device'])
        self.sense_model.to(self.device)
        self.diff_model.to(self.device)

        self.diff_classifier = TextClassificationPipeline(
            model=self.diff_model, tokenizer=self.diff_tokenizer, top_k=None, device=self.diff_model.device)
        # self.diff_classifier = pipeline("text-classification", model=config['diff_model'], top_k=None, device=self.diff_model.device)


    def get_candidates(self, target_word: str, context_sentence, max_hits = 20000):
        """Gets candidates from the index.
        input: 
        query: the string
        context_sentence
        output:
        results: list of id
        query: the actual query used (debug)
        query_doc: the tokenized query (spacy)"""
        result_ids, query, query_doc = self.index.search_term(target_word)
        # take only first 20k sentences to avoid taking too long - some have >170000 results
        results = self.df.loc[result_ids[:max_hits]].copy().reset_index()
        # add the context sentence in front
        context_sentence_row = {"sentence" : [context_sentence]}
        results = pd.concat([pd.DataFrame(context_sentence_row), results]).reset_index(drop=True)
        # reset index to int values because nan was here
        results.loc[0, 'index'] = -1
        results['index'] = results['index'].astype(int)
        return results, query, query_doc
    
    def get_difficulty_level(self, sentences):
        """Call the model to produce a series of annotated values
        input: sentences (str)
        output: list of label strings (N5-N1)"""
        # go into the difficulty classification pipeline
        # use a dataset because its more efficient 
        sentences_hf = Dataset.from_dict({'sentence': sentences})
        diff_preds = self.diff_classifier(sentences_hf['sentence'])
        labels = []
        # collect the argmax
        for diff_pred in diff_preds:
            max_item = max(diff_pred, key=lambda x: x['score'])
            highest_label = max_item['label']
            labels.append(highest_label)
        return labels


    def get_difficulty_level_score(self, levels, target_level, penalty=0.2, higher_level_penalty = 0.4):
        """the score will be one if at the required difficulty level, and -0.2 for each level of difference.
        Maybe if the harder get a harder penalty. like 0.3 vs 0.1
        input: levels, target level: list or single 'N1', 'N2' ... 'N5'
        output: a list of float scores for each input"""
        levels_general = self.label2id.keys()
        level_to_score = {}
        for level in levels_general:
            # using alphabetical ordering for harder
            if level < target_level:
                difference = abs(ord(level[1]) - ord(target_level[1]))
                level_to_score[level] = max(0.0, 1.0 - difference * higher_level_penalty)
            else:
                # numerical ordering trick
                difference = ord(level[1]) - ord(target_level[1])
                level_to_score[level] = max(0.0, 1.0 - difference * penalty)  # Adjust 0.2 as needed for rate of decrease
        result = [level_to_score[level] for level in levels]
        
        return result

    def get_sense_score(self, sentences, target_word):
        # need to have the original sentence also, in last position 
        #sentences_and_context = list(sentences)+[self.nlp.tokenizer(context_sentence)]
        #print(sentences_and_context)
        similarity_matrix, embeddings = utils.get_sense_similarity(
            sentences, target_word, self.nlp,
            self.sense_model, self.sense_tokenizer)
        # now need to return scores comparing the similarity of the context 
        # with all the other sentences
        del embeddings
        return similarity_matrix[0:][0]


    def sort_df_by_keep_first(self, df, by: str, ascending=False):
        # sorting the last stuff
        first_row_df = df.iloc[0:1]
        rest_of_df = df.iloc[1:]

        # Sorting the rest of the DataFrame
        sorted_rest_of_df = rest_of_df.sort_values(by=by, ascending=ascending)

        # Concatenating the first row back
        final_sorted_df = pd.concat([first_row_df, sorted_rest_of_df])

        # Resetting the index if necessary
        final_sorted_df.reset_index(drop=True, inplace=True)
        return final_sorted_df

    def greedy_collect(self, candidates, top_x = 50, K=5):
        final_list_ids = [0]
        
        # not considering the first context sentence
        short_list = candidates.loc[:top_x].copy()
        #print(short_list, type(short_list))

        # preprocess
        short_list['sentence_docs'] = list(self.nlp.pipe(short_list['sentence']))
        short_list['parse_tree'] = [utils.nltk_spacy_tree(sentence, self.nlp, already_parsed=True) 
                                    for sentence in short_list['sentence_docs']]
        short_list['tokenized'] = [[token.text for token in sent] for sent in short_list['sentence_docs']]

        # Then, we build a set of $K$ final sentences using a greedy algorithm, 
        # by including the context sentence as the first 'dummy' element, 
        # then adding to the list the sentence (among the top $X$ quality-ranked candidates) 
        # which also has the highest diversity score (lexical and syntactic).
        for k in range(min(K, len(short_list))):
            # consider each element
            for i in range(0, min(len(short_list),top_x+1)):
                #print(i)
                #temp_list = pd.concat([final_list + short_list.loc[i]])
                #print(temp_list)
                # skip if id is already in the list
                if i in final_list_ids:
                    continue
                # temp_list = final_list.copy()
                # temp_list.append(candidates.loc[i])
                # comp_ids = [el.name for el in temp_list] # name is the old index
                temp_ids = final_list_ids + [i]
                
                # compute scores
                syntax_div_score = 1 - utils.compute_fastkassim_similarity(
                    short_list.loc[temp_ids, 'parse_tree'], self.nlp)
                
                perc_unique_ngrams, _ = utils.compute_distances_and_entropy(
                    short_list.loc[temp_ids,'tokenized'],
                    ngrams=[1, 2, 3, 4])
                lexical_div_score = np.mean(perc_unique_ngrams)
                div_score = 0.5 * syntax_div_score + 0.5 * lexical_div_score
                # assign scores
                short_list.loc[i,'syntax_div_score'] = syntax_div_score
                short_list.loc[i,'lexical_div_score'] = lexical_div_score
                short_list.loc[i,'div_score'] = div_score
                short_list.loc[i,'total_score'] = 0.5 * short_list.loc[i, 'div_score'] + 0.5 * short_list.loc[i, 'quality_score']

            # find the best scoring this round and add it, excluding other already in the list
            only_included = ~short_list.index.isin(final_list_ids)
            filtered_list = short_list[only_included]
            best_pre = filtered_list['total_score'].argmax()
            best = filtered_list.iloc[best_pre].name
            final_list_ids.append(best)

        return short_list.loc[final_list_ids], short_list



    def get_sentence_list(self, target_word, context_sentence, candidates=None, k=5, target_level='N5', **kwargs):
        """If passing candidates, make sure that the context sentence is in first position."""
        # get index matches
        if candidates is None:
            candidates, query, query_doc = self.get_candidates(target_word, context_sentence, max_hits=20000)
            index_hits = len(candidates)
            if index_hits == 0:
                # return empty dataframe
                return pd.DataFrame({"sentence": []})
            candidates.loc[0,'index_hits'] = index_hits
        # sentences = candidates['sentence']

        # get level labels
        candidates['level'] = self.get_difficulty_level(candidates['sentence'])

        # scoring functions for quality here
        # difficulty
        candidates['level_score'] = self.get_difficulty_level_score(levels=candidates['level'], target_level=target_level)

        # reduce candidates to 10k or 5k based on level score
        top_x = 5000
        candidates = self.sort_df_by_keep_first(candidates, 'level_score')[:top_x+1]

        # the context sentence is the first so...
        sentences_docs = list(self.nlp.pipe(candidates['sentence']))
        context_sentence_doc = sentences_docs[0]
        # sense score
        candidates['sense_score'] = self.get_sense_score(sentences_docs, target_word)
        # quality score
        candidates['quality_score'] = 0.5 * candidates['level_score'] + 0.5 * candidates['sense_score']
        
        quality_sorted_candidates = self.sort_df_by_keep_first(candidates, 'quality_score')

        # functions for diversity here
        final_sorted_candidates, short_list = self.greedy_collect(quality_sorted_candidates)

        # save for checking
        final_sorted_candidates.to_csv(f'{self.save_dir}/{target_word}_{target_level}_.csv', index=False)
        short_list.to_csv(f'{self.save_dir}/debug/{target_word}_{target_level}_short_list.csv', index=False)
        
        # k+1 because it contains also the context sentence
        return final_sorted_candidates, short_list