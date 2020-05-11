#!/bin/bash
filedir=$1
lang=$2

if [ "$lang" == "en" ]; then
    #python tokenizer.py --input "${filedir}en.txt" --output "${filedir}en_tok.txt" --lang=en --keep-empty-lines
    python punctuation_remover.py --input "${filedir}en_tok.txt" --output "${filedir}en_tok_punc.txt"
else
    python tokenizer.py --input "${filedir}fr.txt" --output "${filedir}fr_tok.txt" --lang=fr --keep-case --keep-empty-lines
fi
