import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from embedding.BaseEmbedding import BaseEmbedding
import globals as globals_vars
import json
import os


class OneHot(BaseEmbedding):
    def __init__(self, lang, config_path=None):
        if config_path is None:
            config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

        super().__init__(config_path)  # self.config created in super()
        self.model = None
        self.lang = lang

        dict_to_hash = {
            "embedding": self.config["embedding_model"][self.lang],
            "preprocessing_unaligned": self.config["preprocessing"]["raw_unaligned"][
                self.lang
            ],
            "preprocessing_aligned": self.config["preprocessing"]["raw_aligned"][
                self.lang
            ],
            "max_obs": self.config["preprocessing"].get("max_obs", None),
        }
        self.config_hash = super().hash_config(dict_to_hash)

        if os.path.isdir(globals_vars.PATH_SERVER) and globals_vars.TRAINING:

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

    def save_model(self):
        super().save_model(os.path.join(self.path_cache, self.config_hash))

    def fit(self, df):
        super().fit(df, self.lang)

    def _load_model(self):
        super()._load_model(os.path.join(self.path_cache, self.config_hash))

    def _df_to_tensor(self, df):
        tensor = [token for sent in df.text.values.tolist() for token in sent.split()]
        return tensor

    def generate(self, df):
        self._load_model()
        sentences = []
        for sent in df.text.values.tolist():
            if type(sent) == float:
                sent = ""
            sent_tok = [
                self.word_to_idx.get(tok, self.word_to_idx["<UNK>"])
                for tok in sent.split()
            ]
            sentences.append(sent_tok)
        return sentences


###### TESTING #######
if __name__ == "__main__":

    from preprocessing.TxtPreprocessor import run_txt_preprocessor

    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"
    config = {}
    if config_path:
        assert os.path.isfile(config_path), f"invalid user config file: {config_path}"
        with open(config_path) as fd:
            config = json.load(fd)

    lang = "en"

    df_en = run_txt_preprocessor(
        {"preprocessing": config["preprocessing"]}, "raw_aligned", "en"
    )
    df_en = df_en.append(
        run_txt_preprocessor(
            {"preprocessing": config["preprocessing"]}, "raw_unaligned", "en",
        )
    )
    print("head df_en AFTER:\n ", df_en.head(100))

    emb = OneHot("en")
    print(emb.generate(df_en)[:10])
