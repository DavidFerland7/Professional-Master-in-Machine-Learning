import random
import youtokentome as yttm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import globals as globals_vars
import pandas as pd

lang = "en"
align = "raw_unaligned"
suffix_train = "_tok_punc_comb"
train_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    align,
    # lang + config[align][lang].get("data_suffix","") + ".txt"
    lang + suffix_train + ".txt",
)

lang = "en"
align = "raw_unaligned"
suffix_test = "_tok_punc_comb"
test_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    align,
    # lang + config[align][lang].get("data_suffix","") + ".txt"
    lang + suffix_test + ".txt",
)

# print(train_data_path)
# train_data_path = "train_data.txt"
model_path = "example.model"


def load_txt_into_df(filename):
    list_sent = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            list_sent.append(line.rstrip("\n"))
    return pd.DataFrame({"text": list_sent})


# Generating random file with training data
# 10000 lines with 100 characters in each line
n_lines = 10
n_characters = 100
# with open(train_data_path, "w", encoding='utf-8') as fout:
for _ in range(n_lines):
    print("".join([random.choice("abcd ") for _ in range(n_characters)]))

# Generating random text
test_text_old = "".join([random.choice("abcde ") for _ in range(100)])
print("test_text (old): ", test_text_old)


# Training model
model = yttm.BPE.train(
    data=train_data_path,
    model=model_path,
    vocab_size=242,
    bos_id=3,
    eos_id=2,
    coverage=1.0,
)

print("vocab: \n", model.vocab())

# Loading model
bpe = yttm.BPE(model=model_path)

# Two types of tokenization
# print(bpe.encode(test_text, output_type=yttm.OutputType.ID))

# test data
df = load_txt_into_df(test_data_path)[:100]
print(df.head())
# df.loc[:, "text"] = df['text'].apply(lambda x: add_BOS(x))
# df.loc[:, "text"] = df['text'].apply(lambda x: add_EOS(x))

test_text = df.text.values.tolist()
print("test_text: \n", test_text)

enc = bpe.encode(test_text, output_type=yttm.OutputType.SUBWORD, bos=True, eos=True)
enc_back_to_df_for_w2v = pd.DataFrame({"text": [" ".join(x) for x in enc]})
print("enc_back_to_df_for_w2v:\n", enc_back_to_df_for_w2v)

return_from_model = bpe.encode(
    test_text, output_type=yttm.OutputType.ID, bos=True, eos=True
)
print("return_from_model:\n", return_from_model)
voc_data = [x for sent in return_from_model for x in sent]
print(len(voc_data))
print(len(set(voc_data)))
print(set(voc_data))

print("\nvocab_size from obj: \n", bpe.vocab_size())
print("\nvocab_size from obj: \n", bpe.vocab())

convert_back_to_subword = bpe.decode(return_from_model, ignore_ids=[1, 2, 3])
print("convert_back_to_subword:\n", convert_back_to_subword)
