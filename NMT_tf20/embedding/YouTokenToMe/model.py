import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from embedding.BaseEmbedding import BaseEmbedding
import json
import os
import numpy as np
import globals as globals_vars
import youtokentome as yttm
import shutil


class YouTokenToMe(BaseEmbedding):
    def __init__(self, lang, config_path=None):
        if config_path is None:
            config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

        super().__init__(config_path)  # self.config created in super()
        self.lang = lang
        print("\nEmbedding config for lang ({})\n{}".format(self.lang, self.config))

        dict_to_hash = {
            "embedding": self.config["embedding_model"][self.lang],
        }
        self.config_hash = super().hash_config(dict_to_hash)

        if os.path.isdir(globals_vars.PATH_SERVER):

            parent_dir_name = os.path.join(
                os.path.split(
                    os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                )[-1],
                os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1],
            )
            self.path_cache = os.path.join(
                globals_vars.PATH_SERVER, parent_dir_name, self.lang
            )
        else:
            self.path_cache = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), self.lang
            )

    def fit(self):
        # super().fit(df, self.lang)
        train_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "data",
            "raw_unaligned",
            self.lang
            + self.config["embedding_model"][self.lang].get("data_suffix", "")
            + ".txt",
        )
        print("train_data_path: ", train_data_path)

        path_saved_model = os.path.join(self.path_cache, self.config_hash)
        print(
            f"Embedding: Creating {self.__class__.__name__} cache directory with path: ",
            path_saved_model,
        )
        os.makedirs(os.path.join(path_saved_model), exist_ok=True)

        try:
            self.model = yttm.BPE.train(
                data=train_data_path,
                model=os.path.join(path_saved_model, "model.bin"),
                **self.config["embedding_model"][self.lang]["train"],
            )
        # if training did not succeed, delete folder so that it does not try
        # to load it when re-trying in a later run
        except:
            shutil.rmtree(path_saved_model)
            raise ValueError(
                "Training of YouTokenToMe failed for language {}, hash folder will be deleted".format(
                    self.lang
                )
            )

        with open(os.path.join(path_saved_model, "config.json"), "w") as f:
            json.dump(self.config, f)

    def _load_model(self):
        path_saved_model = os.path.join(self.path_cache, self.config_hash)
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
                    os.path.dirname(os.path.abspath(__file__)), "train.py",
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

        self.model = yttm.BPE(
            model=os.path.join(self.path_cache, self.config_hash, "model.bin")
        )

        # already contain '<PAD>', '<UNK>', '<EOS>', '<BOS>'  (no +1 necessary)
        self.vocab_size = self.model.vocab_size()
        print(
            "\nvocab_size from lang {}: \n{}".format(self.lang, self.model.vocab_size())
        )
        print("\nvocab from lang {}: \n{}".format(self.lang, self.model.vocab()))

    def generate(self, df):
        self._load_model()
        sequences_indices = self.model.encode(
            df.text.values.tolist(),
            output_type=yttm.OutputType.ID,
            **self.config["embedding_model"][self.lang]["encode"],
        )
        return sequences_indices

    def convert_idx_to_word(
        self,
        idx_list,
        filepath=None,
        left_trim=None,
        right_trim=None,
        only_write_sent_to_file=False,
    ):
        if not only_write_sent_to_file:
            sentences = self.model.decode(
                idx_list, **self.config["embedding_model"][self.lang]["decode"]
            )
        else:
            sentences = idx_list
        if filepath:

            with open(filepath, "w", encoding="utf-8") as f:
                for sent in sentences:
                    f.write(sent + "\n")
        return sentences
