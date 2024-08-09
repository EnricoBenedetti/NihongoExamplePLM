Writes sentences shorter than 100 to a file, according to those criteria:


- Sentences between 5 and 50 tokens.
- Contain no more than 20% punctuation [sangawa et al]
- Contain no more than 20% digits [sangawa et al]
- Do not contain latin characters. [sangawa et al]
- End in punctuation, and the last word is an adjective, verb, or auxilliary. (this should make the sentences complete, since in Japanese they should end with a copula, verb, or adjective). [a bit different from sangawa et al]


note that the filter can be modified in the file "read_wiki_extracted.py"


Requires (spacy, re, some other stuff)

run 

    build_wikipedia.sh

in the current directory

Version used: https://dumps.wikimedia.org/jawiki/20231201/jawiki-20231201-pages-articles-multistream.xml.bz2

Building the file takes ~5h plus the download time.