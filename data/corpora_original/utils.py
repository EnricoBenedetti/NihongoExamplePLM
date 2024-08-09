import re 
import spacy 
def is_good(sentence, token_limit=50, punct_ratio=0.2, numeral_ratio=0.2, tokenizer=None):
    """Basic surface level requirements for sentence inclusion. requires spacy.
    If a list of tokens is passed, the doc is retrieved.
    If a sentence is passed, it will be tokenized if also the spacy tokenizer is passed, otherwise it will raise exception..."""
    # based on english characters and urls
    # based on having at least 5 tokens [sangawa paper] and ending in punctuation, and ending with a ADJ, VERB, AUX.
    
    if isinstance(sentence, list):
        # retrieve the doc from a list of tokens
        sentence = sentence[0].doc
    elif isinstance(sentence, str):
        sentence = tokenizer(sentence)

    sentence_length = len(sentence) # in tokens!!
    if sentence_length < 5 or sentence_length > token_limit:
        # print('len')
        return False
    
    if (sentence[-1].pos_ != 'PUNCT') or (sentence[-2].pos_ not in ['AUX', 'ADJ', 'VERB']):
    # print('ending')
        return False
    # pattern = r'[a-zA-Z]|https?:\/\/\S+'
    # if re.search(pattern, sentence.text) is not None:
    #     return False
    
    # no more than 20% punctuation or numerals
    #punct_match = re.findall(r'[!\"#$%&\'()*+,-./:;<=>?@[\\\]^_``{|}~…]', sentence.text)
    #punct_count = len(punct_match)
    # we do not consider the last
    punct_count = sum([token.is_punct for token in sentence[0:-1]])
    if punct_count / sentence_length > punct_ratio:
        #print('too much punct',sentence_length, punct_count, punct_count / sentence_length)
        return False
    
    # num_match = re.findall(r'\d', sentence.text)
    # num_count = len(num_match)
    num_count = sum([token.is_digit for token in sentence]) # digit actually means a full token of digits
    if num_count / sentence_length > numeral_ratio:
        #print('too much num',sentence_length, punct_count, punct_count / sentence_length)
        return False
    # no text in other languages
    if re.search(r'.*[A-Za-z].*', sentence.text):
        # print('english')
        return False

    return True