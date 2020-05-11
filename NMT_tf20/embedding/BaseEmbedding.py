import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import globals as globals_vars
import json
import wandb
import hashlib
from collections import Counter


class BaseEmbedding:
    def __init__(self, config_path):
        if not os.path.isfile(config_path):
            raise ValueError(f"invalid user config file: {config_path}")
        with open(config_path) as fd:
            self.config = json.load(fd)
        if wandb.run is not None:
            for k, v in self.config.items():
                wandb.run.summary["embedding_config_" + k] = v

    def generate(self, df, eos_token=False):
        raise NotImplementedError()

    def convert_idx_to_word(
        self,
        idx_list,
        filepath=None,
        left_trim=None,
        right_trim=None,
        only_write_sent_to_file=False,
    ):
        sentences = []
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                for sent in idx_list:
                    # when only_write_sent_to_file==True, it is already a sentence and properly formatted
                    # we still need to trim it after though
                    if not only_write_sent_to_file:
                        # convert idx to word + remove padding
                        sent = " ".join(
                            [self.idx_to_word[str(idx)] for idx in sent if idx != 0]
                        )
                    # trim left and right
                    line_trimmed = self._trim_sentence(
                        sent, left_trim=left_trim, right_trim=right_trim
                    )

                    sentences.append(line_trimmed)
                    f.write(line_trimmed + "\n")
        else:
            for sent in idx_list:
                # when only_write_sent_to_file==True, it is already a sentence and properly formatted
                # we still need to trim it after though
                if not only_write_sent_to_file:
                    # convert idx to word + remove padding
                    sent = " ".join(
                        [self.idx_to_word[str(idx)] for idx in sent if idx != 0]
                    )
                # trim left and right
                line_trimmed = self._trim_sentence(
                    sent, left_trim=left_trim, right_trim=right_trim
                )

                sentences.append(line_trimmed)
        return sentences

    def _trim_sentence(self, sentence, left_trim=None, right_trim=None):
        return self.trim_sentences(
            [sentence], left_trim=left_trim, right_trim=right_trim
        )[0]

    def trim_sentences(self, sentences, left_trim=None, right_trim=None):
        if left_trim is not None:
            for i in range(len(sentences)):
                seq = sentences[i]
                if left_trim in seq:
                    sentences[i] = seq[seq.find(left_trim) + len(left_trim) :].lstrip()
                else:
                    sentences[i] = ""

        if right_trim is not None:
            for i in range(len(sentences)):
                seq = sentences[i]
                if right_trim in seq:
                    sentences[i] = seq[: seq.find(right_trim)].rstrip()
        return sentences

    def fit_vocabulary(self, df, lang):
        corpus = [
            token
            for sent in df.text.values.tolist()
            for token in sent.split()
            if token not in self.config["embedding_model"][lang]["extra_tokens"]
        ]
        count = list(
            list(
                zip(
                    *Counter(corpus).most_common(
                        self.config["embedding_model"][lang]["vocab_size"]
                    )
                )
            )[0]
        )
        for tok in self.config["embedding_model"][lang]["extra_tokens"][::-1]:
            count.insert(0, tok)

        self.word_to_idx, self.idx_to_word = {}, {}
        for i, k in enumerate(count, start=1):
            self.word_to_idx[k] = i
            self.idx_to_word[i] = k
        # +1 since we start idx at 1
        self.vocab_size = len(list(self.word_to_idx.keys())) + 1

    def fit(self, df, lang):
        self.fit_vocabulary(df, lang)

    def save_model(self, path_saved_model):

        print(
            f"Embedding: Creating {self.__class__.__name__} cache directory with path: ",
            path_saved_model,
        )
        os.makedirs(os.path.join(path_saved_model), exist_ok=True)
        with open(os.path.join(path_saved_model, "word_to_idx.json"), "w") as f:
            json.dump(self.word_to_idx, f)

        with open(os.path.join(path_saved_model, "idx_to_word.json"), "w") as f:
            json.dump(self.idx_to_word, f)

        with open(os.path.join(path_saved_model, "config.json"), "w") as f:
            json.dump(self.config, f)

    def hash_config(self, dict_to_hash):
        config_hash = hashlib.md5(json.dumps(dict_to_hash).encode("utf-8")).hexdigest()
        if wandb.run is not None:
            wandb.run.summary["embedding_config_"] = config_hash
        return config_hash

    def _load_model(self, path_saved_model):
        if not os.path.isdir(path_saved_model):
            if not globals_vars.TRAINING:
                raise Exception(
                    f"Embedding: While running in TEST mode: Model is not trained with this config yet \n({path_saved_model})"
                )
            else:
                print(
                    f"Embedding: While running in TRAINING mode: Model is not trained with this config yet -> Now training the model with this config \n({path_saved_model})"
                )

                train_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    self.__class__.__name__,
                    "train.py",
                )
                training_return = os.system(f"python {train_path}")
                if training_return != 0:
                    raise Exception(
                        f"EMBEDDING TRAINING HAS FAILED OUCH\npython {train_path} failed"
                    )
        else:
            print(
                f"Embedding: loading model already trained with config: \n({path_saved_model})\n"
            )
        with open(os.path.join(path_saved_model, "word_to_idx.json"), "r") as f:
            self.word_to_idx = json.load(f)

        with open(os.path.join(path_saved_model, "idx_to_word.json"), "r") as f:
            self.idx_to_word = json.load(f)
        # +1 since we start idx at 1
        self.vocab_size = len(list(self.word_to_idx.keys())) + 1
