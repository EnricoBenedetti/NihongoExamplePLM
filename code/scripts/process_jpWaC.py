import sys
import os
import gzip
import pandas as pd
import re
import argparse
from utils import get_logger


def extract_sentence(input_str):
    """Processes the column formulation into the original sentence."""
    char_list = input_str.split('\n')
    sentence_chars = [char.split('\t')[0] for char in char_list]
    return "".join(sentence_chars)

def process_data_jpwac(file_name):
    
    df = None

    file_name = os.path.realpath(file_name)
    dir_name = os.path.dirname(file_name)

    logger.debug(f'opening {file_name}')
    with gzip.open(file_name, 'rt') as file:
        # read whole file, may take a while
        df = pd.read_xml(file)
    
    df = df.drop(columns=['domain', 'gap'])

    # get the level from filename
    base_name = os.path.basename(file_name)
    match = re.search(r'(\d+)', base_name)
    if match:
        annotated_level = match.group(1)
    else:
        annotated_level = None
    logger.debug(annotated_level)
    df['level_old_jlpt'] = annotated_level

    df_processed = df
    df_processed['sentence'] = df['s'].map(extract_sentence)
    df_processed = df_processed.drop(columns='s')

    # saving to file
    df_base_name = base_name.split('.')[0] + '.csv'
    logger.debug(df_base_name)
    df_file_name = os.path.join(dir_name, df_base_name)
    logger.info(f'saving to {df_file_name}')
    
    df_processed.to_csv(df_file_name)

    logger.debug(f'columns are {df_processed.columns.values}')
    
    return df_processed

# should run with --files ../../data/jpWaC/jpWaC-L*.gz


if __name__ == "__main__":
    logger = get_logger()
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process a list of files")

    # Add an argument to accept a list of files (input from the command line)
    parser.add_argument("--files", nargs="+", help="List of files to process. Should be .tgz.")

    # Parse the command line arguments
    args = parser.parse_args()
    logger.debug(args)

    # process all files passed
    for file in args.files:
        try:
            logger.info(f'processing {file}')
            process_data_jpwac(file)
        except Exception as e:
            logger.error(e.args[0])
            pass
    logger.info('finished')
