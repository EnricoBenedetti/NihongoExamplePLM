# from openai import OpenAI
# nohup python generation.py > generation_py.nohup.log 2>&1 &

# Use a pipeline as a high-level helper
from transformers import TextGenerationPipeline, AutoModelForCausalLM, AutoTokenizer
from utils import *
import utils
import pandas as pd
import numpy as np

def experiment_multiple_ret_sequences(pipe, prompt, target_word, nlp):
    """This experiment runs generation with the given prompt by collecting the outputs of 5 independent beam search generation, and selecting only sentences
    that contain the target word. We do this to counterbalance the difficulty of 0 shot generation."""
    generated_texts = pipe(prompt)
    generated_text = "".join([output['generated_text'] for output in generated_texts])
    #print(f"generated text: {generated_text}")
    candidates = utils.extract_sentences_with_target_word(target_word, generated_text, nlp, clean=True)
    return candidates

def prompt_1shot(target_word, context_sentence, target_level, k):
    prompt = f"""write {k} {target_level} example sentences in japanese, that must contain the word "{target_word}" used in a similar sense as in the initial sentence "{context_sentence}".
example:
input: target word: "する", initial sentence: "自分が自分の心でするべき事だ。" level: beginner
output:
here are the sentences using "する"
1.ビジネスサイクルを実践しましょう。
2.今年も仲良くしてくださいね～。
3.今度はあなたが決断する時です。
4.偏見はなくしたいと思います。
5.私は押し入れの片付けをよくします。
input: target word: "{target_word}", initial sentence: "{context_sentence}" level: {target_level}
output:
here are the sentences using "{target_word}":
1."""
    return prompt

def main():
    utils.fix_reproducibility()
    nlp = spacy.load("ja_ginza")
    model_name = "llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0" # -> prompt actually works for this one
    # model_name = "llm-jp/llm-jp-13b-instruct-full-jaster-v1.0"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # this could use an argument parser

    k = 5
    device = 6
    gen_parameters = {
        # length constraints
        'max_new_tokens' : 300,
        #'min_new_tokens': 5, # does not work
        # gen strategy
        'do_sample' : True, # is false then no sampling but greedy decoding aka get the most frequent
        # 'num_beam_groups': 2,
        'num_beams' : 3,
        'use_cache': True,
        # logits parameters
        'temperature' : 1.0, # the higher the more unconfident the model will be
        'renormalize_logits' : True,
        'top_k' : 50, 
        'top_p' : 1,
        #'epsilon_cutoff' : 3e-4, # does not work
        'repetition_penalty' : 5.0,
        'force_words_ids': None, # this may be the most important cause we want that word to appear...
        # output forms
        'num_return_sequences': 5, # or K?
        'return_full_text' : False
    }
    
    pipe = TextGenerationPipeline(model, tokenizer, device=device, **gen_parameters)


    # run on everything
    dev_only = False
    log = utils.get_logger()
    df_target = pd.read_csv('../../data/targets/target_words.csv')
    #df_target_test = df_target[df_target['is_test'] == False][:5]
    df_target_test = df_target
    k = 5
    target_levels = ['beginner', 'intermediate', 'advanced']
    target_levels_real = ['N5', 'N3', 'N1']
    candidates_len = []
    for target_level, target_level_real in zip(target_levels, target_levels_real):
        for i, data in df_target_test.iterrows():

            target_word = data['target_word']
            context_sentence = data['context_sentence']
            prompt = f'write {k} {target_level} example sentences in japanese, that must contain the word "{target_word}" used in a similar sense as "{context_sentence}". following are {k} diverse sentences that must use "{target_word}": '
            candidates_list = [context_sentence]
            cycle_max = 0
            while len(candidates_list) -1 < k and cycle_max < 10:
                candidates_list += experiment_multiple_ret_sequences(pipe, prompt, target_word, nlp)
                candidates = pd.DataFrame({'sentence': candidates_list})
                candidates.drop_duplicates(subset='sentence', inplace=True)
                candidates_list = candidates['sentence'].to_list()
                cycle_max += 1
            # remove duplicates....
            #print(candidates_list)
            #save file
            candidates['target_word'] = target_word
            candidates['context_sentence'] = context_sentence
            candidates['target_level'] = target_level_real
            candidates_len.append(len(candidates_list)-1)
            log.info(f"{target_word}, {target_level}, {len(candidates_list)-1}, {cycle_max}")
            if len(candidates_list) -1 < k:
                log.error(f"{target_word}, {target_level} has only {len(candidates_list) -1} sentences out of {k}")
            if not dev_only:
                candidates.to_csv(f"/data/enrico_benedetti/nihongoexample/evaluation/outputs/generation/llm_jp/{target_word}_{target_level_real}_.csv", index=False)
            else:
                candidates.to_csv(f"/data/enrico_benedetti/nihongoexample/evaluation/outputs/experiments/llmjp_{target_word}_{target_level_real}_.csv", index=False)

    log.debug(f"Avg number of candidates {np.mean(candidates_len)}, min: {np.min(candidates_len)}")


if __name__ == '__main__':
    main()