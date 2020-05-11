import pandas as pd
from joblib import dump, load
import hashlib
import json
import os
import copy
from pathlib import Path
#import spacy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import globals as globals_vars

## Helper functions
def load_txt_into_df(filename):
    list_sent = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            list_sent.append(line.rstrip("\n"))
    return pd.DataFrame({"text": list_sent})


def write_df_to_txt(filename, df):
    list_sent = df.text.values.tolist()
    with open(filename, "w", encoding="utf-8") as f:
        for l in list_sent:
            f.write("%s\n" % l)


## TxtPreprocessor object
class TxtPreprocessor:
    def __init__(self, lang):

        self.lang = lang

        # spacy make a time-out connection when launching from compute node
        # Commented out the following for the evaluation (and dry-run)
        # if lang == "en":
        #     try:
        #         self.tokenizer = spacy.load(
        #             "en_core_web_sm", spacy_func_to_disable={"tagger", "parser", "ner"}
        #         )
        #     except OSError:
        #         try:
        #             os.system("python -m spacy download en_core_web_sm")
        #         except Exception:
        #             raise ValueError(
        #                 "Failed to download en spacy model (maybe no internet connection?)"
        #             )
        # elif lang == "fr":
        #     try:
        #         self.tokenizer = spacy.load(
        #             "fr_core_news_sm", spacy_func_to_disable={"tagger", "parser", "ner"}
        #         )
        #     except OSError:
        #         try:
        #             os.system("python -m spacy download fr_core_news_sm")
        #         except Exception:
        #             raise ValueError(
        #                 "Failed to download fr spacy model (maybe no internet connection?)"
        #             )
        # else:
        #     raise ValueError("lang {} not supported".format(lang))

    ## NOTE: if you ever change a function here that was already implemented.
    # Please notify everyone and everyone has to delete the cache in its home directory and on team server.

    def add_EOS(self, s):
        return " ".join(s.split(" ") + ["<EOS>"])

    def add_BOS(self, s):
        return " ".join(["<BOS>"] + s.split(" "))

    def lower(self, s):
        return s.lower()

    def to_unicode(self, s):
        raise NotImplementedError()

    def strip_tags(self, s):
        raise NotImplementedError()

    def strip_punctuation(self, s):
        raise NotImplementedError()

    def strip_multiple_whitespaces(self, s):
        raise NotImplementedError()

    def strip_numeric(self, s):
        raise NotImplementedError()

    def remove_stopwords(self, s):
        raise NotImplementedError()

    def strip_short(self, s):
        raise NotImplementedError()

    def stem_text(self, s):
        raise NotImplementedError()

    # def spacy_tokenizer(self, s):
    #     return " ".join([token.text for token in self.tokenizer(s)])

    # To be applied on string
    def apply_on_raw_strings(self, raw_strings, task_to_apply_txt_pp_ops_on):

        # execute requested(train) / cached(test) operations
        for txt_pp_ops, args, kwargs in task_to_apply_txt_pp_ops_on:
            if args and kwargs:
                raw_strings = raw_strings.apply(
                    lambda s: getattr(self, txt_pp_ops)(s, args, kwargs)
                )
            elif args:
                raw_strings = raw_strings.apply(
                    lambda s: getattr(self, txt_pp_ops)(s, args)
                )
            elif kwargs:
                raw_strings = raw_strings.apply(
                    lambda s: getattr(self, txt_pp_ops)(s, kwargs)
                )
            else:
                raw_strings = raw_strings.apply(lambda s: getattr(self, txt_pp_ops)(s))

        return raw_strings


def run_txt_preprocessor(config, align, lang, dst_path=None, path_input_txt=None):
    config = config["preprocessing"]

    if not globals_vars.TRAINING:
        path_cache = dst_path
    else:
        if os.path.isdir(globals_vars.PATH_SERVER):
            parent_dir_name = os.path.dirname(os.path.abspath(__file__)).split("/")[
                -1
            ]  # e.g.: preprocessing
            path_cache = os.path.join(
                globals_vars.PATH_SERVER, parent_dir_name, align, lang
            )
        elif dst_path:
            # dst_path is assumed to be a relative path (or will crash) (or not we really don't know)
            path_cache = os.path.join(dst_path, align, lang)
        else:
            path_cache = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), align, lang
            )

    dict_to_hash = {
        "preprocessing": config[align][lang],
    }
    dict_to_hash.update({"max_obs": config.get("max_obs", None)})
    config_hash = hashlib.md5(json.dumps(dict_to_hash).encode("utf-8")).hexdigest()

    # TRAINING Mode : cache for requested config already exists
    if globals_vars.TRAINING and os.path.isdir(os.path.join(path_cache, config_hash)):
        print(
            f"Preprocessing ({align}-{lang}): Cache directory already exists with hash: ",
            str(config_hash),
        )
        df_train, df_valid = None, None

        if os.path.isfile(os.path.join(path_cache, config_hash, "train.txt")):
            df_train = load_txt_into_df(
                os.path.join(path_cache, config_hash, "train.txt")
            )
        if os.path.isfile(os.path.join(path_cache, config_hash, "valid.txt")):
            df_valid = load_txt_into_df(
                os.path.join(path_cache, config_hash, "valid.txt")
            )

    # TRAINING Mode : cache for requested config does NOT exits
    elif globals_vars.TRAINING:
        print(
            f"Preprocessing ({align}-{lang}): Creating cache directory with hash: ",
            str(config_hash),
        )
        df = load_txt_into_df(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "data",
                align,
                lang + config[align][lang].get("data_suffix", "") + ".txt",
            )
        )
        df = df[: config.get("max_obs", None)]
        txt_preprocessor = TxtPreprocessor(lang)

        def _df_preprocess(df, filename):
            if "ops_on_raw_strings" in config[align][lang]:
                df.loc[:, "text"] = txt_preprocessor.apply_on_raw_strings(
                    df["text"], config[align][lang]["ops_on_raw_strings"]
                )
            os.makedirs(os.path.join(path_cache, config_hash), exist_ok=True)
            write_df_to_txt(os.path.join(path_cache, config_hash, filename), df)
            # df.to_csv(os.path.join(path_cache, config_hash, filename))
            return df

        if config[align][lang].get("split", 1) < 1:
            n_train = int(df.shape[0] * config[align][lang]["split"])
            df_train = _df_preprocess(df.iloc[:n_train], "train.txt")
            df_valid = _df_preprocess(df.drop(df_train.index), "valid.txt",)
        else:
            df_train = _df_preprocess(df, "train.txt")
            df_valid = None
        with open(
            os.path.join(path_cache, config_hash, "config_preprocessing.json"), "w"
        ) as f:
            json.dump(config, f)

    # TEST Mode : txt input comes from file at path_input_txt
    else:
        print("Preprocessing: TEST Mode : txt input comes from file at path_input_txt")
        df = load_txt_into_df(path_input_txt)
        df_train = df
        df_valid = None
        txt_preprocessor = TxtPreprocessor(lang)

        if "ops_on_raw_strings" in config[align][lang]:
            df_train.loc[:, "text"] = txt_preprocessor.apply_on_raw_strings(
                df_train["text"], config[align][lang]["ops_on_raw_strings"]
            )

    # Return 1 or 2 df depending on config[align][lang]["split"]
    if df_valid is not None:
        return df_train, df_valid
    else:
        return df_train


if __name__ == "__main__":

    user_config_path = "config.json"
    user_config = {}
    if user_config_path:
        assert os.path.isfile(
            user_config_path
        ), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    align = "raw_unaligned"
    lang = "en"
    df_input_DONOTUSE = pd.read_table(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            align,
            lang + ".txt",
        ),
        header=None,
        names=["text"],
    )
    print("head df_input_DONOTUSE BEFORE:\n{}\n".format(df_input_DONOTUSE.tail(10)))

    df_train, df_valid = run_txt_preprocessor(user_config, align, lang)
    print("head df_valid AFTER:\n ", df_valid.tail(10))
