import os
import random
import numpy as np
from transformers import set_seed
import logging
import sys
import spacy 
import nltk
from nltk import Tree
from nltk.translate.bleu_score import sentence_bleu
from typing import Union
from itertools import combinations
import fkassim.FastKassim as fkassim
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import re

RNG_SEED = 42

# Create a logger
def get_logger():
    """Returns logger for the script"""

    program_name = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program_name)

    # Set the logging level (choose one of: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create a file handler to save logs to a file
    file_handler = logging.FileHandler(f'{program_name}logs.log')

    # Create a formatter to specify the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger


def fix_reproducibility():
   # Set a seed value
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(RNG_SEED)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(RNG_SEED)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(RNG_SEED)
    set_seed(RNG_SEED)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

### SPAN AND OTHER SPACY UTILITIES

def get_spans(query, sentence, tokenizer, verbose=False):
    """input:
    query: the target word to find the span for. (if already a doc it should be faster)
    sentence: the sentence. (can be a doc - it will be faster)
    tokenizer: the tokenizer. assumed a direct __call__ method (e.g. from spacy tokenizers)

    output: 
    spans_ids_char, spans_ids_spacy: two lists of lists of tuples
    each list of tuples [(s, e)_i]: list of start and end idx of matches. one can take only the first one if needed.
    if no match is found, a tuple of empty lists is returned."""

    matcher = spacy.matcher.Matcher(tokenizer.vocab)
    # tokenize 
    if isinstance(sentence, spacy.tokens.doc.Doc):
        sentence_tokens = sentence
    else:
        sentence_tokens = tokenizer(sentence) # this is prob the heavy one

    if isinstance(query, spacy.tokens.doc.Doc):
        query_tokens = query
    else:
       query_tokens = tokenizer(query) # this is prob the heavy one - shorter tho
    
    pattern = [ {"LEMMA": query_token.lemma_} for query_token in query_tokens]
    matcher.add("query_match", [pattern])

    # get matches as id, start, end
    matches = matcher(sentence_tokens)
    # delete this particular one to avoid matching later
    matcher.remove("query_match")

    results_char = []
    results_spacy = []
    for match_id, start, end in matches:
        span_doc = sentence_tokens[start:end]
        #span_str = sentence_tokens.text[span_doc.start_char:span_doc.end_char]
        # if verbose:
        #     print(f'sentence: {sentence}')
        #     print(f'Span(token): {sentence_tokens[start:end]} , Span(str): {span_str}')
        #     print(f'str: {span_doc.start_char,span_doc.end_char} , indeces: {start, end}')
        results_char.append((span_doc.start_char, span_doc.end_char))
        results_spacy.append((start, end))

    return results_char, results_spacy

# https://www.mrklie.com/post/2020-09-26-pretokenized-bert/
def tokenize_for_sense_embeddings(pretokenized_sentence, tokenizer):
    """Processes a pretokenized sentence
    Input:
    tokenizer: the hf tokenizer"""
    grouped_inputs = [torch.LongTensor([tokenizer.cls_token_id])]
    subtokens_per_token = []

    for token in pretokenized_sentence:
        tokens = tokenizer.encode(
            token,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True
        ).squeeze(axis=0)
        grouped_inputs.append(tokens)
        subtokens_per_token.append(len(tokens))

    grouped_inputs.append(torch.LongTensor([tokenizer.sep_token_id]))

    flattened_inputs = torch.cat(grouped_inputs)
    flattened_inputs = torch.unsqueeze(flattened_inputs, 0)
    return flattened_inputs, grouped_inputs, subtokens_per_token

def get_subtoken_span(subtokens_per_token, span_spacy):
    """Gets the start and end of subtokens from the spacy span."""
    target_word_start, target_word_end = span_spacy
    return sum(subtokens_per_token[:target_word_start]), sum(subtokens_per_token[:target_word_end])

def get_embeddings(flattened_inputs, subtoken_spans, model, layer_start = 9, layer_end = 13, method='mean'):
    """Gets the embeddings from the model on the average of layers. 

    Input: 
    flattened_inputs: the tokenized (hf) ids, expects a batch.
    subtoken_span: the corresponding subtoken spans to extract. also a batch.
    layer_start, layer_end: which layers to average embeddings.
    method: first, mean, sum. does not change much
    
    output:
    a batch of final embeddings for the subtoken_spans which correspond to the target word."""
    # the memory problem is probably here in putting on cuda everything...
    device = model.device
    output = model(input_ids=flattened_inputs['input_ids'].to(device),attention_mask=flattened_inputs['attention_mask'].to(device), output_hidden_states=True)
    hidden_states = output.hidden_states # the first number is the layer, second is the token number, last is the vector
    average_layer_batch = sum(hidden_states[layer_start:layer_end]) / (layer_end-layer_start) # we get the token mean across the last layers
    sentence_num_in_batch = average_layer_batch.size()[0]
    
    out_tensors = []

    for num in range(sentence_num_in_batch): # for all sentences passed
    # here we need the start/end on subtokens
    ###
        #print(average_layer_batch[num].size()) 
        sentence_embeddings = average_layer_batch[num] # get a tensor num_tokens(hf) x 768 vector values
        #print(sentence_embeddings[0].shape) # sentence_embeddings[i] is the embedding of subtoken i. to get the ones for the original words we need to...
        sentence_embeddings_no_st = sentence_embeddings[1:-1]# remove first and last (they are the cls and sep ones) - ok theres also the pad but they are right-side
        #print(sentence_embeddings_no_st.shape) # find the embeddings of interest (the one for a certain target word) and sum? also between them / or take the first one?
        
        subtoken_start, subtoken_end =  subtoken_spans[num] # get the current spans / the one of the batch
        target_word_embeddings = sentence_embeddings_no_st[subtoken_start:subtoken_end] # those are the corresponding embeddings
        if method=='sum':
            target_word_emb_final = target_word_embeddings.sum(0)
        elif method=='mean':
            target_word_emb_final = target_word_embeddings.mean(0)
        elif method=='first':
            target_word_emb_final = target_word_embeddings[0] #take the first token only

        out_tensors.append(target_word_emb_final)

    out_tensor = torch.stack(out_tensors, dim=0).detach().cpu().numpy()
    return out_tensor

def get_sense_similarity(sentences: list, target_word, nlp, model, tokenizer, method='mean'):
    """Returns the similarity matrix and the embeddings.
    If a list of docs is passed, it will be faster"""
    if not isinstance(sentences[0], spacy.tokens.doc.Doc):
        pretoken_sentences = list(nlp.pipe(sentences))
    else:
        pretoken_sentences = sentences
    pretoken_sentences = [[t.orth_ for t in doc ]for doc in pretoken_sentences]
    sentence_spans = []
    subtoken_spans = []
    batch_flattened_inputs = []
    # list of indeces to zero the distances for in the matrix
    i_zero_list = []
    for i, (sentence, pretoken_sentence) in enumerate(zip(sentences, pretoken_sentences)):
        # process tokens in the batch
        # these 2 for every sentence in the batch?
        flattened_inputs, grouped_inputs, subtokens_per_token = tokenize_for_sense_embeddings(pretokenized_sentence=pretoken_sentence, tokenizer=tokenizer)
        batch_flattened_inputs.append(flattened_inputs.tolist()[0]) # depack the batch

        spans_ids_char, spans_ids_spacy = get_spans(query=target_word, sentence=sentence, tokenizer=nlp.tokenizer)
        # if empty, remember the sentence and set the distance to 0 later. give a 0,0 subtoken span
        if len(spans_ids_char) == 0:
            subtoken_span = (0,0)
            i_zero_list.append(i)
        else:
            # do it normally
            spans_ids_char = spans_ids_char[0] # take first occurrence
            spans_ids_spacy = spans_ids_spacy[0] # take first occurrence
            sentence_spans.append(spans_ids_spacy)  
            #print(sentence_spans)
            #print(subtokens_per_token)
            subtoken_span = get_subtoken_span(subtokens_per_token, spans_ids_spacy)
        subtoken_spans.append(subtoken_span)

    # need to pad and stuff...
    #print(batch_flattened_inputs)
    batch_size = 16
    embeddings = []
    input_batches = [batch_flattened_inputs[i:i + batch_size] for i in range(0, len(batch_flattened_inputs), batch_size)]
    subtoken_spans_batches = [subtoken_spans[i:i + batch_size] for i in range(0, len(subtoken_spans), batch_size)]
   
    for input, sub_spans_batch in zip(input_batches, subtoken_spans_batches):

        # fix stupid bug where a sentence gets tokenized into floats(?)
        try:
            in_x = tokenizer.batch_encode_plus(input, is_split_into_words=True, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
        except ValueError:
            input = [np.array(tokens).astype(int).tolist() for tokens in input]
            in_x = tokenizer.batch_encode_plus(input, is_split_into_words=True, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')

        batch_embeddings = get_embeddings(flattened_inputs=in_x, subtoken_spans=sub_spans_batch, model=model, method=method)
        embeddings.append(batch_embeddings)
    
    embeddings = torch.Tensor(np.vstack(embeddings))
    # maybe need to put them into a tensor before?
    similarity_matrix = pairwise_cosine_similarity(embeddings, zero_diagonal=True)

    # for the parts where there are no target words, fill with -1s
    similarity_matrix[:, i_zero_list] = -1
    similarity_matrix[i_zero_list, :] = -1
    # re_zero diagonal
    similarity_matrix.fill_diagonal_(0)

    return similarity_matrix, embeddings

### DIVERSITY UTILS

def get_tokenized_sentences(sentences: str, tokenizer) -> list:
    """returns a list tokenized sentences (list of list of str)"""
    return [[token.text for token in tokenizer(sent)] for sent in sentences]

def nltk_spacy_tree(sent, nlp, already_parsed=False):
    """
    Visualize the SpaCy dependency tree with nltk.tree
    """
    if already_parsed == False:
        doc = nlp(sent)
    else:
        doc = sent
    def token_format(token):
        return "_".join([token.pos_, token.dep_])

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(token_format(node),
                       [to_nltk_tree(child) 
                        for child in node.children]
                   )
        else:
            return token_format(node)

    tree = [to_nltk_tree(sent.root) for sent in doc.sents]
    # The first item in the list is the full tree
    return tree[0]


# now we need to do comparisons for lists of sentences
def compute_fastkassim_similarity(input: list, nlp):
    """Given a a list of sentences or already computed trees, will produce the trees and perform pairwise comparisons and take the avg"""
    FastKassim = fkassim.FastKassim(fkassim.FastKassim.LTK)
    if isinstance(input[0], nltk.Tree):
        s_trees = input
    else:
        s_trees = []
        for s in input:
            s_trees.append(nltk_spacy_tree(s, nlp=nlp))

    sims = []

    for s1, s2 in combinations(s_trees,2):
        sims.append(FastKassim.compute_similarity_preparsed(s1,s2))
    return np.mean(sims)

def compute_self_BLEU(tokenized_sentence_list, weights=[0.25, 0.25, 0.25, 0.25]):
    """Computes BLEU score considering each sentence as hypotesis and the others as references, then averages the scores.
    input: the list of sentences,
    weights: the weights to give n-grams
    ---
    output: the bleu score."""

    scores = []
    for i in range(len(tokenized_sentence_list)):
        hypothesis = tokenized_sentence_list[i]
        references = tokenized_sentence_list[:i] + tokenized_sentence_list[i+1:]
        score = sentence_bleu(hypothesis=hypothesis, references=references, weights=weights)
        scores.append(score)
    return np.mean(scores)

# this function's code is from he et al. https://github.com/NLPCode/CDEG/blob/master/evaluations/automatic_evaluation.py
def compute_distances_and_entropy(tokenized_sentences_list, ngrams=[1, 2, 3, 4], num_tokens=0):
        """
        this function is used to calculate the percentage of unique n-grams to measure the generation diversity.
        this function is also used to calculate entropy to measure the generation diversity.
        :param tokenized_sentences_list:
        :param ngrams:
        :param num_tokens:
        :return: the percentage of unique n-grams and entropy of their distribution
        """
        distances = []
        entropies = []
        if num_tokens > 0:
            cur_num = 0
            new_tokenized_sentences_list = []
            for tokenized_sentence in tokenized_sentences_list:
                cur_num += len(tokenized_sentence)
                new_tokenized_sentences_list.append(tokenized_sentence)
                if cur_num >= num_tokens:
                    break
            tokenized_sentences_list = new_tokenized_sentences_list

        for n in ngrams:
            # calculate (n-gram, frequency) pairs
            ngram_fdist = nltk.FreqDist()
            for tokens in tokenized_sentences_list:
                ngrams = nltk.ngrams(tokens, n)
                ngram_fdist.update(ngrams)
            unique = ngram_fdist.B()  # the number of unique ngrams
            total = ngram_fdist.N()  # the number of ngrams
            distances.append(unique * 1.0 / total)
            # calculate entropies
            ans = 0.0
            for k, v in ngram_fdist.items():
                ans += v * np.log(v * 1.0 / total)
            ans = -ans / total
            entropies.append(ans)
        return distances, entropies

## GENERATION UTILS

def extract_sentences_with_target_word(target_word: str, text: str, nlp, clean=True):
    """From text, sentencize and returns a list of only sentences that contain the target word according to the utils.get_spans function.
    If clean is set to true, will eliminate numbers and punctuation form the beginning."""
    # Process the text
    doc = nlp(text)
        
    # Tokenize the text into sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    garbage_start_pattern = r"^[\d.,() ]+"

    # print(sentences)
    sentences_word_ok = []
    for sentence in sentences:
        if clean:
        # remove numbers and stuff
            sentence = re.sub(garbage_start_pattern, '', sentence).strip()
        spans_char, spans_spacy = get_spans(query=target_word, sentence=sentence, tokenizer=nlp.tokenizer)
        # get the list of sentences that contain the word
        if len(spans_char) > 0:
            sentences_word_ok.append(sentence)
            #print(spans_char)
    return sentences_word_ok

## EVAL utils

def get_sentences_block(df, block_id, sys_random_ordering):
    """Gets a sub dataframe for a block id and the system (according to the random ordering)"""
    sheet_mapping = {1: "A", 2: "E", 3: "I"}
    return df[(df['block_id']== block_id) & (df['random_ordering'] == sheet_mapping[sys_random_ordering])]

def get_meta_info_block(combined_df, block_id):
    """Returns target word, target level and context sentence for one block"""
    temp = get_sentences_block(combined_df, block_id, 1)
    target_word = temp.iloc[0]['target_word']
    target_level = temp.iloc[0]['target_level']
    context_sentence = temp.iloc[0]['context_sentence']
    return target_word, target_level, context_sentence

def get_full_block_info(combined_df, block_id):
    result = {}
    target_word, target_level, context_sentence = get_meta_info_block(combined_df, block_id)
    result['target_word'] = target_word
    result['target_level'] = target_level
    result['context_sentence'] = context_sentence
    result['s1'] = get_sentences_block(combined_df, block_id, 1)
    result['s2'] = get_sentences_block(combined_df, block_id, 2)
    result['s3'] = get_sentences_block(combined_df, block_id, 3)
    return result

def format_sentences_gpt(sentences):
    return "\n".join(sentences)

def __init__():
    fix_reproducibility()