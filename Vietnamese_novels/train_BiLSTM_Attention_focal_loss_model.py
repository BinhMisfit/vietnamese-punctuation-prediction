from preprocessing import process_data
from model import BiLSTM_model, batchnize_dataset, BiLSTM_CRF_model,BiLSTM_Attention_model
# dataset path
raw_path = 'dataset/Cleansed_data'
save_path = "dataset/Encoded_data"
# embedding path
Word2vec_path = "embeddings"

char_lowercase = True
# dataset for train, validate and test
vocab = "dataset/Encoded_data/vocab.json"
train_set = "dataset/Encoded_data/train.json"
dev_set = "dataset/Encoded_data/dev.json"
test_set = "dataset/Encoded_data/test.json"
word_embedding = "dataset/Encoded_data/word_emb.npz"
# network parameters
num_units = 300
emb_dim = 300
char_emb_dim = 52
filter_sizes = [25, 25]
channel_sizes = [5, 5]
# training parameters
lr = 0.001
lr_decay = 0.05
minimal_lr = 1e-5
keep_prob = 0.5
batch_size = 32
epochs = 30
max_to_keep = 1
no_imprv_tolerance = 20
checkpoint_path = "checkpoint_BiLSTM_Att/"
summary_path = "checkpoint_BiLSTM_Att/summary/"
model_name = "punctuation_model"

config = {"raw_path": raw_path,\
          "save_path": save_path,\
          "Word2vec_path":Word2vec_path,\
          "char_lowercase": char_lowercase,\
          "vocab": vocab,\
          "train_set": train_set,\
          "dev_set": dev_set,\
          "test_set": test_set,\
          "word_embedding": word_embedding,\
          "num_units": num_units,\
          "emb_dim": emb_dim,\
          "char_emb_dim": char_emb_dim,\
          "filter_sizes": filter_sizes,\
          "channel_sizes": channel_sizes,\
          "lr": lr,\
          "lr_decay": lr_decay,\
          "minimal_lr": minimal_lr,\
          "keep_prob": keep_prob,\
          "batch_size": batch_size,\
          "epochs": epochs,\
          "max_to_keep": max_to_keep,\
          "no_imprv_tolerance": no_imprv_tolerance,\
          "checkpoint_path": checkpoint_path,\
          "summary_path": summary_path,\
          "model_name": model_name}

# alpha & gamma for focal loss (tune hyperparameter)
alpha = 0.1
gamma = 0.5
import os
if not os.path.exists(config["save_path"]):
    os.mkdir(config["save_path"])
    process_data(config)

print("Load datasets...")
# used for training
train_set = batchnize_dataset(config["train_set"], config["batch_size"], shuffle=True)
# used for computing validate loss
valid_set = batchnize_dataset(config["dev_set"], batch_size=100, shuffle=False)

import tensorflow as tf
tf.reset_default_graph()
print("Build models...")
model = BiLSTM_Attention_model(config, alpha, gamma)
model.train(train_set, valid_set)
# used for computing test precision, recall and F1 scores
test_set = batchnize_dataset(config["test_set"], batch_size=100, shuffle=False)
# run the session
model.restore_last_session(checkpoint_path)
model.test(test_set)
