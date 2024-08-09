# download the dump

wget https://dumps.wikimedia.org/jawiki/20231201/jawiki-20231201-pages-articles-multistream.xml.bz2
# below does not work. https://github.com/attardi/wikiextractor/issues/222
#python wikiextractor/WikiExtractor.py -o extracted --json jawiki-20231201-pages-articles

pip install wikiextractor
pip install nltk
pip install sudachipy sudachidict-full

python -m wikiextractor.WikiExtractor -o extracted --json jawiki-20231201-pages-articles-multistream.xml.bz2
# make the text file, one sentence per line. takes around ~5hours
python read_wiki_extracted.py --extracted_dir extracted --token_limit 50
# for nohup background processing
# nohup python read_wiki_extracted.py --extracted_dir extracted --token_limit 50 &
# rm -r extracted