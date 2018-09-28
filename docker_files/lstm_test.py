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

'''Import dev file TO GUARANTEE ALL TAGS IN TESTING'''
#collection = open(sys.argv[2], 'r')
#collection = open("dev_all.txt", 'r')
#collection = open("test_all.txt", 'r')
collection = open("bigtest_all.txt", 'r')
dev_data = collection.read()
collection.close()

'''Prepare test data into a list of sentences'''
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

print(len(tags_index))

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
num_epochs = 10
####################

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


''' JUST WRITTEN DOMAINS '''
# Number of epochs:   50
# overall accuracy: 0.6898004028566197
# unknown word accuracy: 0.3548531719345438

''' ALL DOMAINS '''
# Number of epochs:   10
# overall accuracy: 0.7532901897920007
# unknown word accuracy: 0.3613335274421313

# not lowercased
# 10 epochs
# overall accuracy: 0.7504799224207881
# unknown word accuracy: 0.2202826267664173

# all lowercased
# batch size = 256
# 10 epochs
# overall accuracy: 0.7566545943913396
# unknown word accuracy: 0.21691013722886232


# all lowercased
# batch size = 200
# 10 epochs
# overall accuracy: 0.7589503057650062
# unknown word accuracy: 0.22465692784417884



'''
batch size = 256
epoch: 1
100%|████████████████████████████████████████████| 21/21 [07:36<00:00, 21.72s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.70s/it]
overall accuracy: 0.24857012804528092
unknown word accuracy: 0.06750774679061532
epoch: 2
100%|████████████████████████████████████████████| 21/21 [07:40<00:00, 21.93s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.71s/it]
overall accuracy: 0.48362326584733517
unknown word accuracy: 0.16135458167330677
epoch: 3
100%|████████████████████████████████████████████| 21/21 [07:39<00:00, 21.88s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.5813493241504878
unknown word accuracy: 0.17928286852589642
epoch: 4
100%|████████████████████████████████████████████| 21/21 [07:40<00:00, 21.91s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.602841932355677
unknown word accuracy: 0.20340858787073926
epoch: 5
100%|████████████████████████████████████████████| 21/21 [07:37<00:00, 21.79s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.72s/it]
overall accuracy: 0.649864434285262
unknown word accuracy: 0.2098273572377158
epoch: 6
100%|████████████████████████████████████████████| 21/21 [07:40<00:00, 21.92s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.71s/it]
overall accuracy: 0.6847948702725167
unknown word accuracy: 0.22266489597166889
epoch: 7
100%|████████████████████████████████████████████| 21/21 [07:37<00:00, 21.77s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.70s/it]
overall accuracy: 0.7044271606404243
unknown word accuracy: 0.21204072598494908
epoch: 8
100%|████████████████████████████████████████████| 21/21 [07:36<00:00, 21.76s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.7086029804666627
unknown word accuracy: 0.23749446657813192
epoch: 9
100%|████████████████████████████████████████████| 21/21 [07:38<00:00, 21.84s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.70s/it]
overall accuracy: 0.740129430623998
unknown word accuracy: 0.2647189021691014
epoch: 10
100%|████████████████████████████████████████████| 21/21 [07:48<00:00, 22.33s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.70s/it]
overall accuracy: 0.7508361534960122
unknown word accuracy: 0.23306772908366533
epoch: 11
100%|████████████████████████████████████████████| 21/21 [07:40<00:00, 21.95s/it]
100%|████████████████████████████████████████████| 22/22 [02:50<00:00,  7.73s/it]
overall accuracy: 0.7590096776108769
unknown word accuracy: 0.25387339530765823
epoch: 12
100%|████████████████████████████████████████████| 21/21 [07:37<00:00, 21.80s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.7701715846345663
unknown word accuracy: 0.23395307658255865
epoch: 13
100%|████████████████████████████████████████████| 21/21 [07:39<00:00, 21.86s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.70s/it]
overall accuracy: 0.7725068772388134
unknown word accuracy: 0.24302788844621515
epoch: 14
100%|████████████████████████████████████████████| 21/21 [07:39<00:00, 21.89s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.71s/it]
overall accuracy: 0.7727245740070059
unknown word accuracy: 0.2107127047366091
epoch: 15
100%|████████████████████████████████████████████| 21/21 [07:39<00:00, 21.90s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.71s/it]
overall accuracy: 0.7773555779849195
unknown word accuracy: 0.21491810535635236
epoch: 16
100%|████████████████████████████████████████████| 21/21 [07:38<00:00, 21.85s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.7876664885511291
unknown word accuracy: 0.22532093846834883
epoch: 17
100%|████████████████████████████████████████████| 21/21 [07:39<00:00, 21.90s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.70s/it]
overall accuracy: 0.7831938094955372
unknown word accuracy: 0.20053120849933598
epoch: 18
100%|████████████████████████████████████████████| 21/21 [07:36<00:00, 21.75s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.7903580122306002
unknown word accuracy: 0.25143868968570166
epoch: 19
100%|████████████████████████████████████████████| 21/21 [07:36<00:00, 21.76s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.7907340339211146
unknown word accuracy: 0.22620628596724215
epoch: 20
100%|████████████████████████████████████████████| 21/21 [07:35<00:00, 21.71s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.7933661857547151
unknown word accuracy: 0.21447543160690571
epoch: 21
100%|████████████████████████████████████████████| 21/21 [07:37<00:00, 21.80s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.7963941498941202
unknown word accuracy: 0.24435590969455512
epoch: 22
100%|████████████████████████████████████████████| 21/21 [07:38<00:00, 21.84s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.72s/it]
overall accuracy: 0.7916641928397554
unknown word accuracy: 0.2067286409915892
epoch: 23
100%|████████████████████████████████████████████| 21/21 [07:33<00:00, 21.62s/it]
100%|████████████████████████████████████████████| 22/22 [02:48<00:00,  7.68s/it]
overall accuracy: 0.7882008351639652
unknown word accuracy: 0.20185922974767595
epoch: 24
100%|████████████████████████████████████████████| 21/21 [07:42<00:00, 22.02s/it]
100%|████████████████████████████████████████████| 22/22 [02:48<00:00,  7.68s/it]
overall accuracy: 0.7975222149656633
unknown word accuracy: 0.2036299247454626
epoch: 25
100%|████████████████████████████████████████████| 21/21 [07:40<00:00, 21.93s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.69s/it]
overall accuracy: 0.7964535217399908
unknown word accuracy: 0.1952191235059761
epoch: 26
100%|████████████████████████████████████████████| 21/21 [07:40<00:00, 21.93s/it]
100%|████████████████████████████████████████████| 22/22 [02:49<00:00,  7.71s/it]
overall accuracy: 0.7966910091234737
unknown word accuracy: 0.2058432934926959
epoch: 27
100%|████████████████████████████████████████████| 21/21 [07:40<00:00, 21.94s/it]
100%|████████████████████████████████████████████| 22/22 [03:03<00:00,  8.36s/it]
overall accuracy: 0.8013022224860971
unknown word accuracy: 0.21602478972996902
epoch: 28
100%|████████████████████████████████████████████| 21/21 [07:44<00:00, 22.12s/it]
100%|████████████████████████████████████████████| 22/22 [02:48<00:00,  7.67s/it]
overall accuracy: 0.797838864810307
unknown word accuracy: 0.21248339973439576
epoch: 29
100%|████████████████████████████████████████████| 21/21 [07:41<00:00, 21.99s/it]
100%|████████████████████████████████████████████| 22/22 [02:50<00:00,  7.75s/it]
overall accuracy: 0.799560648340557
unknown word accuracy: 0.19278441788401948
epoch: 30
100%|████████████████████████████████████████████| 21/21 [07:39<00:00, 21.87s/it]
100%|████████████████████████████████████████████| 22/22 [02:48<00:00,  7.67s/it]
overall accuracy: 0.8033406558609907
unknown word accuracy: 0.22841965471447542

'''



