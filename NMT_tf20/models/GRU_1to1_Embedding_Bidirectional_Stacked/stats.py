import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import importlib
import pandas as pd
import fire
import string

model_name = os.path.dirname(os.path.abspath(__file__)).split("/")[
    -1
]  # e.g.: GRU_1to1_Onehot

################# UTILS FUNCTIONS ###################
from preprocessing.TxtPreprocessor import load_txt_into_df, write_df_to_txt

print("string.punctuation: ", string.punctuation)


def count_punc(s, chars_to_count, exclude_last_char=True):
    ### character by character ###
    # removing <UNK> because otherwise the < > are recognized as punctuation
    # print("\ns:", s)
    # s_temp = s.replace('<UNK> ','')
    # print("s_temp:", s_temp)
    # if exclude_last_char:
    #     func = lambda l1,l2: sum([1 for x in l1[:-2] if x in l2])
    # else:
    #     func = lambda l1,l2: sum([1 for x in l1 if x in l2])
    # return func(s_temp, chars_to_count)

    # token by token
    if exclude_last_char:
        return sum([1 for x in s.split()[:-1] if x in chars_to_count])
    else:
        return sum([1 for x in s.split() if x in chars_to_count])


def run_stats(run_name):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(current_file_dir, run_name, "stats")

    df = pd.read_csv(os.path.join(stats_path, "raw_data.csv"), index_col=0)
    ################# MODIFY HERE TO ADD STATS ################

    ### Stats sentence lengths ###
    # Are we doing better or worse on longer sentence, shorter sentence, avg length ?
    df.loc[:, "x_sentences_sent_len"] = df.x_sentences.str.count(" ") + 1
    df.loc[:, "y_target_sent_len"] = df.y_target.str.count(" ") + 1

    # How we doing better or worse on sentences where length of english and french differs?
    df.loc[:, "ratio_y_target_on_x"] = (
        df.loc[:, "y_target_sent_len"] / df.loc[:, "x_sentences_sent_len"]
    )

    # How long are we predicting vs the target length
    # (not necessarly linked with bleu metric, but could look at the impact on bleu of predicting wrong length)
    df.loc[:, "y_pred_sent_len"] = df.y_pred.str.count(" ") + 1
    df.loc[:, "diff_len_y_pred_vs_y_target"] = (
        df.loc[:, "y_pred_sent_len"] - df.loc[:, "y_target_sent_len"]
    )

    # impact on bleu of % of UNK - source english side
    x_sentences_count_UNK = df.x_sentences.str.count("<UNK>")
    df.loc[:, "x_sentences_pct_UNK"] = (
        x_sentences_count_UNK / df.loc[:, "x_sentences_sent_len"]
    )

    # impact on bleu of % of UNK - target french side
    y_target_count_UNK = df.y_target.str.count("<UNK>")
    df.loc[:, "y_target_pct_UNK"] = y_target_count_UNK / df.loc[:, "y_target_sent_len"]

    # impact on bleu of % of punctuation - target french side (excluding last . )
    df.loc[:, "y_target_count_punc"] = df["y_target"].apply(
        lambda s: count_punc(s, string.punctuation, exclude_last_char=True)
    )
    df.loc[:, "y_target_pct_punc"] = (
        df.loc[:, "y_target_count_punc"] / df.loc[:, "y_target_sent_len"]
    )


    ################# MODIFY HERE TO ADD STATS ################
    df.to_csv(os.path.join(stats_path, "stats.csv"))


def main(run_name):
    run_stats(run_name)


if __name__ == "__main__":
    fire.Fire(main)
