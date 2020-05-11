import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy
from solution import RNN, GRU
from joblib import dump, load

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU
from models import make_model as TRANSFORMER


MODEL = "GRU"
SEQ_LEN = 35

# 35 seq-rrn: 1500
# 70 seq-rnn: 1001

# 35 seq-GRU: 950
# 70 seq-GRU: 5

if MODEL == "RNN" and SEQ_LEN == 70:
    lc_path = "RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_0" \
        + "/best_params.pt"
    HARD_CODED_SEED = 550
elif MODEL == "RNN" and SEQ_LEN == 35:
    lc_path = "RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_0" \
        + "/best_params.pt"
    HARD_CODED_SEED = 1500
elif MODEL == "GRU" and SEQ_LEN == 70:
    lc_path = "GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_save_best_0" \
        + "/best_params.pt"
    HARD_CODED_SEED = 5
elif MODEL == "GRU" and SEQ_LEN == 35:
    lc_path = "GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_save_best_0" \
        + "/best_params.pt"
    HARD_CODED_SEED = 950
#lc_path = "3_1_1_RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_0" \
#    + "/best_params.pt"

#lc_path = "RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_0" \
#   + "/best_params.pt"

# lc_path = "GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_save_best_0" \
#     + "/best_params.pt"

#model = np.load(lc_path, allow_pickle=True)

##############################################################################
#
# RETRIEVE VOCABULARY MAPPING
#
##############################################################################

with open('word_to_id_' + '.pickle', 'rb') as f:
    word_to_id = load(f)
vocab_size = len(word_to_id)

id_to_word = {v: k for k, v in word_to_id.items()}

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set. Only do this \
                    ONCE for each model setting, and only after you've \
                    completed ALL hyperparameter tuning on the validation set.\
                    Note we are not requiring you to do this.")

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')


def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# set_seed(args.seed)
set_seed(HARD_CODED_SEED)

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


###############################################################################
#
# MODEL SETUP
#
###############################################################################

# NOTE ==============================================
# This is where your model code will be called.
if args.model == 'RNN':
    model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
elif args.model == 'GRU':
    model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
elif args.model == 'TRANSFORMER':
    if args.debug:  # use a very small model
        model = TRANSFORMER(vocab_size=vocab_size, n_units=16, n_blocks=2)
    else:
        # Note that we're using num_layers and hidden_size to mean slightly
        # different things here than in the RNNs.
        # Also, the Transformer also has other hyperparameters
        # (such as the number of attention heads) which can change it's behavior.
        model = TRANSFORMER(vocab_size=vocab_size, n_units=args.hidden_size,
                            n_blocks=args.num_layers, dropout=1. - args.dp_keep_prob)
    # these 3 attributes don't affect the Transformer's computations;
    # they are only used in run_epoch
    model.batch_size = args.batch_size
    model.seq_len = args.seq_len
    model.vocab_size = vocab_size
else:
    print("Model type not recognized.")


model = model.to(device)
model.load_state_dict(torch.load(lc_path))


#model = torch.load(map_location=torch.device('cpu'))
model.eval()

print(1)
# print(model)

###############################################################################
#
# GENERATE DATA
#
###############################################################################
samples_id_with_inputs = []
samples_id = []

samples_words_with_inputs = []
samples_words = []

inputs = torch.from_numpy(np.random.randint(0, 10000, 128).astype(np.int64)).contiguous().to(device)

hidden = model.init_hidden()[0]
hidden = hidden.to(device)

samples_all = model.generate(inputs, hidden, SEQ_LEN)

print(samples_all[:, :10])
for i in range(10):
    # print(samples_all[:,i])
    # print(samples_all[:,:10].T)
    # print(samples_all[:,:10].T.tolist())
    samples_id_with_inputs.append([inputs[i].tolist()] + samples_all[:, i].T.tolist())
    samples_id.append(samples_all[:, i].T.tolist())

print(len(samples_id_with_inputs))
print(len(samples_id_with_inputs[0]))
# print(samples_id)
# print(samples_id.shape)
for s, sentence in enumerate(samples_id_with_inputs):
    samples_words_with_inputs.append([])
    for token in sentence:
        samples_words_with_inputs[s].append(id_to_word[token])

for s, sentence in enumerate(samples_id):
    samples_words.append([])
    for token in sentence:
        samples_words[s].append(id_to_word[token])

print(samples_words_with_inputs)
print(samples_words)
print(1)
with open('samples_3_' + MODEL + '_seq_' + str(SEQ_LEN) + '_with_inputs.txt', 'w') as f:
    for item in samples_words_with_inputs:
        print([item[0]] + [item[1:]])
        print(" ".join(item[1:]))
        f.write("{}\n".format([item[0], " ".join(item[1:])]))

with open('samples_3_' + MODEL + '_seq_' + str(SEQ_LEN) + '.txt', 'w') as f:
    for item in samples_words:
        f.write("{}\n".format([item[0]] + [item[1:]]))
