import pandas as pd
from joblib import dump, load
import hashlib
import json
import os
import copy
from pathlib import Path
import spacy
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from preprocessing.TxtPreprocessor import load_txt_into_df, run_txt_preprocessor

user_config_path = "config.json"
user_config = {}
if user_config_path:
    assert os.path.isfile(
        user_config_path
    ), f"invalid user config file: {user_config_path}"
    with open(user_config_path, "r") as fd:
        user_config = json.load(fd)

# align = "raw_aligned"
for align in ["raw_aligned", "raw_unaligned"]:
    for lang in ["en", "fr"]:
        # lang = "en"
        path_df = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            align,
            lang + ".txt",
        )
        # ORIGINAL DF
        locals()["df_orig_" + align + "_" + lang] = load_txt_into_df(path_df)
        print(
            "shape 'df_orig_{}_{} BEFORE:\n{}\n".format(
                align, lang, locals()["df_orig_" + align + "_" + lang].shape
            )
        )
        print(
            "head(5) 'df_orig_{}_{} BEFORE:\n{}\n".format(
                align, lang, locals()["df_orig_" + align + "_" + lang].head(5)
            )
        )
        print(
            "tail(5) 'df_orig_{}_{} BEFORE:\n{}\n".format(
                align, lang, locals()["df_orig_" + align + "_" + lang].tail(5)
            )
        )

        # PREPROCESSED DF
        locals()["df_pp_" + align + "_" + lang] = run_txt_preprocessor(
            {"preprocessing": user_config["preprocessing"]},
            align,
            lang,
            path_input_txt=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "data",
                align,
                lang
                + user_config["preprocessing"][align][lang].get("data_suffix", "")
                + ".txt",
            ),
        )

        print(
            "shape 'df_pp_{}_{} AFTER:\n{}\n".format(
                align, lang, locals()["df_pp_" + align + "_" + lang].shape
            )
        )
        print(
            "head(5) 'df_pp_{}_{} AFTER:\n{}\n".format(
                align, lang, locals()["df_pp_" + align + "_" + lang].head(5)
            )
        )
        print(
            "tail(5) 'df_pp_{}_{} AFTER:\n{}\n".format(
                align, lang, locals()["df_pp_" + align + "_" + lang].tail(5)
            )
        )


print("\n####################### STATS ##########################\n")
df_stats = pd.DataFrame(
    {
        "sent_orig_en": df_orig_raw_aligned_en.text,
        "sent_orig_fr": df_orig_raw_aligned_fr.text,
        "sent_pp_en": df_pp_raw_aligned_en.text,
        "sent_pp_fr": df_pp_raw_aligned_fr.text,
        "len_orig_en": df_orig_raw_aligned_en.text.str.count(" ") + 1,
        "len_orig_fr": df_orig_raw_aligned_fr.text.str.count(" ") + 1,
        "len_orig_max_both": np.maximum(
            df_orig_raw_aligned_en.text.str.count(" ") + 1,
            df_orig_raw_aligned_fr.text.str.count(" ") + 1,
        ),
        "len_pp_en": df_pp_raw_aligned_en.text.str.count(" ") + 1,
        "len_pp_fr": df_pp_raw_aligned_fr.text.str.count(" ") + 1,
        "len_pp_max_both": np.maximum(
            df_pp_raw_aligned_en.text.str.count(" ") + 1,
            df_pp_raw_aligned_fr.text.str.count(" ") + 1,
        ),
    }
)

groups = pd.DataFrame(
    {
        "sent_orig_fr": df_stats["sent_orig_fr"],
        "sent_orig_en": df_stats["sent_orig_en"],
        "sent_pp_fr": df_stats["sent_pp_fr"],
        "sent_pp_en": df_stats["sent_pp_en"],
        "len_pp_max_both": df_stats["len_pp_max_both"],
        "len_pp_max_both_range": pd.qcut(
            df_stats["len_pp_max_both"],
            q=list(np.arange(0, 1, 0.04)) + [0.97, 0.98, 0.99, 0.995, 1],
        ),
        "len_pp_max_both_bin_num": pd.qcut(
            df_stats["len_pp_max_both"],
            q=list(np.arange(0, 1, 0.04)) + [0.97, 0.98, 0.99, 0.995, 1],
            labels=False,
        ),
        "len_pp_fr": df_stats["len_pp_fr"],
        "len_pp_fr_range": pd.qcut(
            df_stats["len_pp_fr"],
            q=list(np.arange(0, 1, 0.04)) + [0.97, 0.98, 0.99, 0.995, 1],
        ),
        "len_pp_fr_bin_num": pd.qcut(
            df_stats["len_pp_fr"],
            q=list(np.arange(0, 1, 0.04)) + [0.97, 0.98, 0.99, 0.995, 1],
            labels=False,
        ),
        "len_pp_en": df_stats["len_pp_en"],
        "len_pp_en_range": pd.qcut(
            df_stats["len_pp_en"],
            q=list(np.arange(0, 1, 0.04)) + [0.97, 0.98, 0.99, 0.995, 1],
        ),
        "len_pp_en_bin_num": pd.qcut(
            df_stats["len_pp_en"],
            q=list(np.arange(0, 1, 0.04)) + [0.97, 0.98, 0.99, 0.995, 1],
            labels=False,
        ),
    }
)


os.makedirs("output", exist_ok=True)
groups.to_csv("output/misc_stats.csv", index=False)


#######  EN ORIG ######
print("\n #######  EN ORIG ######")
sent_orig_both_alignment_en = (
    df_orig_raw_aligned_en.text.values.tolist()
    + df_orig_raw_unaligned_en.text.values.tolist()
)

all_toks_orig_both_alignment_en = [
    token
    for sent in sent_orig_both_alignment_en
    for token in sent.split()
    # if token not in self.config["embedding_model"]["extra_tokens"]
]
print(
    "all tokens size both alignement - orig - EN: ",
    len(all_toks_orig_both_alignment_en),
)

voc_orig_both_alignment_en = list(set(all_toks_orig_both_alignment_en))
print(voc_orig_both_alignment_en[:50])
print("Voc size both alignement - orig - EN: ", len(voc_orig_both_alignment_en))

#######  EN PP ######
print("\n #######  EN PP ######")
sent_pp_both_alignment_en = (
    df_pp_raw_aligned_en.text.values.tolist()
    + df_pp_raw_unaligned_en.text.values.tolist()
)

all_toks_pp_both_alignment_en = [
    token
    for sent in sent_pp_both_alignment_en
    for token in sent.split()
    # if token not in self.config["embedding_model"]["extra_tokens"]
]
print("all tokens size both alignement - pp - EN: ", len(all_toks_pp_both_alignment_en))

voc_pp_both_alignment_en = list(set(all_toks_pp_both_alignment_en))
print(voc_pp_both_alignment_en[:50])
print("Voc size both alignement - pp - EN: ", len(voc_pp_both_alignment_en))

#######  FR ORIG ######
print("\n #######  FR ORIG ######")
sent_orig_both_alignment_fr = (
    df_orig_raw_aligned_fr.text.values.tolist()
    + df_orig_raw_unaligned_fr.text.values.tolist()
)

all_toks_orig_both_alignment_fr = [
    token
    for sent in sent_orig_both_alignment_fr
    for token in sent.split()
    # if token not in self.config["embedding_model"]["extra_tokens"]
]
print(
    "all tokens size both alignement - orig - FR: ",
    len(all_toks_orig_both_alignment_fr),
)

voc_orig_both_alignment_fr = list(set(all_toks_orig_both_alignment_fr))
print(voc_orig_both_alignment_fr[:50])
print("Voc size both alignement - orig - FR: ", len(voc_orig_both_alignment_fr))

#######  FR PP ######
print("\n #######  FR PP ######")
sent_pp_both_alignment_fr = (
    df_pp_raw_aligned_fr.text.values.tolist()
    + df_pp_raw_unaligned_fr.text.values.tolist()
)

all_toks_pp_both_alignment_fr = [
    token
    for sent in sent_pp_both_alignment_fr
    for token in sent.split()
    # if token not in self.config["embedding_model"]["extra_tokens"]
]
print("all tokens size both alignement - pp - FR: ", len(all_toks_pp_both_alignment_fr))

token_fr_pp, counts_fr_pp = np.unique(all_toks_pp_both_alignment_fr, return_counts=True)
# array = np.vstack([token_fr_pp, counts_fr_pp]).T
# sorted_array = array[array[:,1].argsort()][::-1]
# token_fr_pp, counts_fr_pp = sorted_array[:,0], sorted_array[:,1]
sort_index_fr_pp = (-counts_fr_pp).argsort()
token_fr_pp, counts_fr_pp = (
    token_fr_pp[sort_index_fr_pp],
    counts_fr_pp[sort_index_fr_pp],
)

print(np.cumsum(counts_fr_pp) / np.sum(counts_fr_pp))
print(token_fr_pp.tolist())
print(counts_fr_pp.tolist())
df_cumsum_fr_pp = pd.DataFrame(
    {
        "token": token_fr_pp.tolist(),
        "freq": counts_fr_pp.tolist(),
        "cum_freq": (np.cumsum(counts_fr_pp) / np.sum(counts_fr_pp)).tolist(),
    }
).sort_values(by="freq", ascending=False)

df_cumsum_fr_pp.to_csv("output/cum_freq_fr_pp.csv")

voc_pp_both_alignment_fr = list(set(all_toks_pp_both_alignment_fr))
print(voc_pp_both_alignment_fr[:50])
print("Voc size both alignement - pp - FR: ", len(voc_pp_both_alignment_fr))

