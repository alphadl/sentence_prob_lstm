from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import numpy as np


def prepare_sentence(seq, maxlen):
    # Pads seq and slides windows
    x = []
    y = []
    for i, w in enumerate(seq):
        x_padded = pad_sequences([seq[:i]],
                                 maxlen=maxlen - 1,
                                 padding='pre')[0]  # Pads before each sequence
        x.append(x_padded)
        y.append(w)
    return x, y


# Data
data = ["Two little dicky birds",
        "Sat on a wall,",
        "One called Peter,",
        "One called Paul.",
        "Fly away, Peter,",
        "Fly away, Paul!",
        "Come back, Peter,",
        "Come back, Paul."]

# Preprocess data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
vocab = tokenizer.word_index
seqs = tokenizer.texts_to_sequences(data)

# Slide windows over each sentence
maxlen = max([len(seq) for seq in seqs])
x = []
y = []
for seq in seqs:
    x_windows, y_windows = prepare_sentence(seq, maxlen)
    x += x_windows
    y += y_windows
x = np.array(x)
y = np.array(y) - 1
y = np.eye(len(vocab))[y]  # One hot encoding

# Define model
model = Sequential()
model.add(Embedding(input_dim=len(vocab) + 1,  # vocabulary size. Adding an
                                               # extra element for <PAD> word
                    output_dim=5,  # size of embeddings
                    input_length=maxlen - 1))  # length of the padded sequences
model.add(LSTM(10))
model.add(Dense(len(vocab), activation='softmax'))
model.compile('rmsprop', 'categorical_crossentropy')

# Train network
model.fit(x, y, epochs=1000)

# Compute probability of occurence of a sentence
sentence = "One called Peter,"
tok = tokenizer.texts_to_sequences([sentence])[0]
x_test, y_test = prepare_sentence(tok, maxlen)
x_test = np.array(x_test)
y_test = np.array(y_test) - 1  # The word <PAD> does not have a class
p_pred = model.predict(x_test)
vocab_inv = {v: k for k, v in vocab.items()}
log_p_sentence = 0
for i, prob in enumerate(p_pred):
    word = vocab_inv[y_test[i]+1]  # Index 0 from vocab is reserved to <PAD>
    history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
    prob_word = prob[y_test[i]]
    log_p_sentence += np.log(prob_word)
    print('P(w={}|h={})={}'.format(word, history, prob_word))
print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
