huggingface-cli login
# paste token write from huggingface
# git clone https://huggingface.co/datasets/bennexx/jp_sentences
cd jp_sentences/
cp ../corpus.csv .
cp ../sources.csv .
git lfs track corpus.csv
git add .gitattributes corpus.csv sources.csv
git commit -m "add corpus"
huggingface-cli lfs-enable-largefiles .
git push