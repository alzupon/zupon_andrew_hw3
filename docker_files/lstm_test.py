#   Andrew Lee Zupon
#   LING 538    Fall 02017
#   FINAL PROJECT   LSTM TESTING
#   Python 3
##### IMPORT PACKAGES #####
import numpy as np
import pickle
import sys
from tqdm import tqdm
import dynet_config
dynet_config.set(
    mem=4096,
    autobatch=True,      # utilize autobatching
    random_seed=1978     # simply for reproducibility here
)
import dynet as dy
##### IMPORT DATA FILES #####
'''Import training file'''
#collection = open(sys.argv[1], 'r')
collection = open("train_all.txt", 'r')
training_data = collection.read()
collection.close()

'''Prepare training data into a list of sentences'''
train_data = training_data.split('\n')
train_data2 = []
for item in train_data:
    item = item.split(' ')
    train_data2.append(item)
train_data3 = []
for item in train_data2:
    temp = []
    for x in item:
        x = x.lower()
        y = x.split("/")
        temp.append(y)
    train_data3.append(temp)
train_sent = []
for sentence in train_data3:
    stemp = [["<s>","<s>"]]
    for item in sentence:
        if len(item) >1:
            stemp.append(item)
        else:
            continue
    train_sent.append(stemp)
train_words = []
train_tags = []
for sentence in train_sent:
    sent_words = []
    sent_tags = []
    for item in sentence:
        w = item[0]
        t = item[-1]
        sent_words.append(w)
        sent_tags.append(t)
    train_words.append(sent_words)
    train_tags.append(sent_tags)
kw = []
for sentence in train_words:
    for word in sentence:
        word2 = word.lower()
        kw.append(word2)

'''Import dev file''' 
#collection = open(sys.argv[2], 'r')
#collection = open("dev_all.txt", 'r')
#collection = open("test_all.txt", 'r')
collection = open("bigtest_all.txt", 'r')
dev_data = collection.read()
collection.close()

'''Prepare dev data into a list of sentences'''
dev_data = dev_data.split('\n')
dev_data2 = []
for item in dev_data:
    item = item.split(' ')
    dev_data2.append(item)
dev_data3 = []
for item in dev_data2:
    temp = []
    for x in item:
        x = x.lower()
        y = x.split("/")
        temp.append(y)
    dev_data3.append(temp)
dev_sent = []
for sentence in dev_data3:
    stemp = [["<s>","<s>"]]
    for item in sentence:
        if len(item) >1:
            stemp.append(item)
        else:
            continue
    dev_sent.append(stemp)
dev_words = []
dev_tags = []
for sentence in dev_sent:
    sent_words = []
    sent_tags = []
    for item in sentence:
        w = item[0]
        t = item[-1]
        sent_words.append(w)
        sent_tags.append(t)
    dev_words.append(sent_words)
    dev_tags.append(sent_tags)

'''Import test file'''
#collection = open(sys.argv[2], 'r')
collection = open("dev_all.txt", 'r')
#collection = open("test_all.txt", 'r')
#collection = open("bigtest_all.txt", 'r')
testing_data = collection.read()
collection.close()

'''Prepare test data into a list of sentences'''
test_data = testing_data.split('\n')
test_data2 = []
for item in test_data:
    item = item.split(' ')
    test_data2.append(item)
test_data3 = []
for item in test_data2:
    temp = []
    for x in item:
        x = x.lower()
        y = x.split("/")
        temp.append(y)
    test_data3.append(temp)
test_sent = []
for sentence in test_data3:
    stemp = [["<s>","<s>"]]
    for item in sentence:
        if len(item) >1:
            stemp.append(item)
        else:
            continue
    test_sent.append(stemp)
test_words = []
test_tags = []
for sentence in test_sent:
    sent_words = []
    sent_tags = []
    for item in sentence:
        w = item[0]
        t = item[-1]
        sent_words.append(w)
        sent_tags.append(t)
    test_words.append(sent_words)
    test_tags.append(sent_tags)


all_tags = train_tags+dev_tags+test_tags


'''indices for tags'''
def tag_to_index(tagslist):
    tag_dict = {"<unk>":0}
    i=1
    for x in tagslist:
        for y in x:
            if y in tag_dict:
                continue
            else:
                tag_dict[y] = i
                i+=1
    return tag_dict


# create indices for all tags
#tags_index = tag_to_index(train_tags)
tags_index = tag_to_index(all_tags)
# link tags in training and testing data to indices
train_tags = [[tags_index[t] for t in sentence] for sentence in train_tags]
dev_tags = [[tags_index[t] for t in sentence] for sentence in dev_tags]
test_tags = [[tags_index[t] for t in sentence] for sentence in test_tags]
# create reverse to look up tags based on index
index_tags = dict((v,k) for k,v in tags_index.items())


''' Set up LSTM '''
LSTM_model = dy.ParameterCollection()
#################
embedding_size = 50
hidden_size = 200
num_layers = 1
#################


##### PREPARE WORD EMBEDDINGS #####
word_index = {"<unk>": 0}
i = 1
for sentence in train_words:
    for word in sentence:
        word = word.lower()
        if word not in word_index:
            i += 1
            word_index[word] = i

'''indices for words'''
def word_to_index(sequence, word_index):
    seq_index = []
    for x in sequence:
        x = x.lower()
        i = word_index.get(x, 0)
        seq_index.append(i)
    return seq_index
index_words = dict((v,k) for k,v in word_index.items())

##### LSTM PARAMETERS #####
embedding_parameters = LSTM_model.add_lookup_parameters((len(word_index)+1, embedding_size))
pW = LSTM_model.add_parameters((hidden_size, len(list(tags_index.keys()))))
pb = LSTM_model.add_parameters((len(list(tags_index.keys()))))
LSTM_unit = dy.LSTMBuilder(num_layers, embedding_size, hidden_size, LSTM_model)

#LSTM_model.populate(sys.argv[3])
LSTM_model.populate("model_baseline")

''' forward pass '''
def forward_pass(x):
    input_sequence = [embedding_parameters[i] for i in x]
    W = dy.parameter(pW)
    b = dy.parameter(pb)
    #initialize rnn
    lstm_seq = LSTM_unit.initial_state()
    lstm_hidden_outputs = lstm_seq.transduce(input_sequence)
    lstm_outputs = [dy.transpose(W) * h + b for h in lstm_hidden_outputs]
    return lstm_outputs

''' prediction function '''
def predict(list_of_outputs):
    # take the softmax of each timestep
    pred_probs = [dy.softmax(o) for o in list_of_outputs]     
    # convert each timestep's output to a numpy array
    pred_probs_np = [o.npvalue() for o in pred_probs]
    # take the argmax for each step
    pred_probs_idx = [np.argmax(o) for o in pred_probs_np]
    return pred_probs_idx


####################
trainer = dy.SimpleSGDTrainer(m=LSTM_model,learning_rate=0.01)
batch_size = 256
num_batches_training = int(np.ceil(len(train_words) / batch_size))
num_batches_testing = int(np.ceil(len(test_words) / batch_size))
num_epochs = 30
####################

mistakes = []
##### TESTING FUNCTION #####
def test():
    correct = 0
    incorrect = 0
    u_correct = 0
    u_incorrect = 0

    for batch in tqdm(range(num_batches_testing)):
        dy.renew_cg()
        # build the batch
        batch_words = test_words[batch*batch_size:(batch+1)*batch_size]
        batch_tags = test_tags[batch*batch_size:(batch+1)*batch_size]
        # iterate through the batch
        for sentence in range(len(batch_words)):
            # prepare input: words to indexes
            seq_of_idxs = word_to_index(batch_words[sentence], word_index)
            # make a forward pass
            preds = forward_pass(seq_of_idxs)
            tag_preds = predict(preds)
            for i in range(len(batch_words[sentence])):
                temp = []
                temp.extend((batch_words[sentence][i],batch_tags[sentence][i],tag_preds[i]))
                #print(temp, temp[0] in kw)
                if temp[1] == temp[2]:
                    correct += 1
                else:
                    incorrect += 1
                    error = batch_words[sentence][i]+"\t"+index_tags.get(temp[1])+"\t"+index_tags.get(temp[2])
                    mistakes.append(error)
                if temp[0] not in kw:
                    if temp[1] == temp[2]:
                        u_correct += 1
                    else:
                        u_incorrect += 1
    overall_accuracy = float(correct)/float(correct+incorrect)
    unknown_accuracy = float(u_correct)/float(u_correct+u_incorrect)
    print("overall accuracy: {}".format(overall_accuracy))
    print("unknown word accuracy: {}".format(unknown_accuracy))
    return overall_accuracy, unknown_accuracy


##### USE THIS TO TEST #####
final_predictions = test()


# if you want to print out the incorrect predictions, sorted by frequency
'''
counts = []
for item in mistakes:
    counts.append(mistakes.count(item))
count_dict = dict(zip(mistakes,counts))
for key, value in sorted(count_dict.items(), key=lambda x: x[1]): 
    print("{} : {}".format(key, value))
    '''
