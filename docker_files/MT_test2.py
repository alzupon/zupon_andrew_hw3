#   Andrew Lee Zupon
#   CSC 585    Fall 02018
#   HW 4   LSTM TESTING
#   Python 3
##### IMPORT PACKAGES #####
import numpy as np
import pickle
import sys
from collections import Counter
from tqdm import tqdm

import dynet_config
dynet_config.set(
    mem=4096,
    autobatch=True,      # utilize autobatching
    random_seed=1978     # simply for reproducibility here
)
import dynet as dy


#### MODEL HYPERPARAMETERS ####
# set these to determine which models to test
num_epochs = 10    # only 10 for now
num_irish = 600    # 0, 500, 600, or 1000
#model_type = "student_bi"
model_type = "teacher_weighted_bi"
#model_type = "teacher_EMA_bi"
###############################


#### Import preprocessed data ####

#import training words
with open('train_words.txt', 'rb') as train_w:
    train_words = pickle.load(train_w)
#import training tags
with open('train_tags.txt', 'rb') as train_t:
    train_tags = pickle.load(train_t)
#import dev words
with open('dev_words.txt', 'rb') as dev_w:
    dev_words = pickle.load(dev_w)
#import dev tags
with open('dev_tags.txt', 'rb') as dev_t:
    dev_tags = pickle.load(dev_t)
#import test words
with open('test_words.txt', 'rb') as test_w:
    test_words = pickle.load(test_w)
#import test tags
with open('test_tags.txt', 'rb') as test_t:
    test_tags = pickle.load(test_t)
#import known words
with open('known_words.txt', 'rb') as known_w:
    known_words = pickle.load(known_w)
#import Irish words
if num_irish != 0:
    with open('irish_words_'+str(num_irish)+'.txt', 'rb') as irish_w:
        irish_words = pickle.load(irish_w)
    #import Irish tags
    with open('irish_tags_'+str(num_irish)+'.txt', 'rb') as irish_t:
        irish_tags = pickle.load(irish_t)

# make short tags
train_tags_short = []
for sent in train_tags:
    temp_sent = []
    for tag in sent:
        try:
            tag = tag[:1]
        except:
            continue
        temp_sent.append(tag)
    train_tags_short.append(temp_sent)
dev_tags_short = []
for sent in dev_tags:
    temp_sent = []
    for tag in sent:
        try:
            tag = tag[:1]
        except:
            continue
        temp_sent.append(tag)
    dev_tags_short.append(temp_sent)
test_tags_short = []
for sent in test_tags:
    temp_sent = []
    for tag in sent:
        try:
            tag = tag[:1]
        except:
            continue
        temp_sent.append(tag)
    test_tags_short.append(temp_sent)
all_tags_short = train_tags_short+dev_tags_short+test_tags_short

'''add the Irish to the training data'''
if num_irish != 0:      
    irish_tags_short = []
    for sent in irish_tags:
        temp_sent = []
        for tag in sent:
            try:
                tag = tag[:1]
            except:
                continue
            temp_sent.append(tag)
        irish_tags_short.append(temp_sent)
    
    # REVERSE for bidirectional (poor man's version)
    train_words2 = train_words[::-1]
    dev_words2 = dev_words[::-1]
    test_words2 = test_words[::-1]
    irish_words2 = irish_words[::-1]
    train_tags_short2 = train_tags_short[::-1]
    dev_tags_short2 = dev_tags_short[::-1]
    test_tags_short2 = test_tags_short[::-1]
    irish_tags_short2 = irish_tags_short[::-1]
    
    # COMBINE for bidirectional (poor man's version)
    train_words = train_words+train_words2
    train_tags_short = train_tags_short+train_tags_short2
    irish_words = irish_words+irish_words2
    irish_tags_short = irish_tags_short+irish_tags_short2
    
    all_tags_short = train_tags_short+dev_tags_short+test_tags_short+irish_tags_short
    train_words = train_words+irish_words
    train_tags_short = train_tags_short+irish_tags_short
    
    # test on dev
    #test_words = dev_words
    #test_tags_short = dev_tags_short

# keep track of tags in training set
tags_collection = []
for sent in all_tags_short:
    for item in sent:
        tags_collection.append(item)


'''indices for tags'''
def tag_to_index(tagslist):
    tag_dict = {"#":0}
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
tags_index = tag_to_index(all_tags_short)

# link tags in to indices
train_tags = [[tags_index[t] for t in sentence] for sentence in train_tags_short]
dev_tags = [[tags_index[t] for t in sentence] for sentence in dev_tags_short]
test_tags = [[tags_index[t] for t in sentence] for sentence in test_tags_short]

# create reverse to look up tags based on index
index_tags = dict((v,k) for k,v in tags_index.items())


''' PREPARE WORD EMBEDDINGS '''
word_index = {"<unk>": 0}
i = 1
for sentence in train_words:
    for word in sentence:
        word = word.lower()
        if word not in word_index:
            i += 1
            word_index[word] = i

# create indices for words
def word_to_index(sequence, word_index):
    seq_index = []
    for x in sequence:
        x = x.lower()
        i = word_index.get(x, 0)
        seq_index.append(i)
    return seq_index
index_words = dict((v,k) for k,v in word_index.items())


''' Set up LSTM '''

LSTM_model = dy.ParameterCollection()
#################
embedding_size = 50
hidden_size = 200
num_layers = 1
#################

##### LSTM PARAMETERS #####
embedding_parameters = LSTM_model.add_lookup_parameters((len(word_index)+1, embedding_size))
W = LSTM_model.add_parameters((hidden_size, len(list(tags_index.keys()))))
b = LSTM_model.add_parameters((len(list(tags_index.keys()))))
LSTM_unit = dy.LSTMBuilder(num_layers, embedding_size, hidden_size, LSTM_model)


#### Populate LSTM with saved parameters ####
mt_model = "models/model_"+"epochs="+str(num_epochs)+"_irish="+str(num_irish)+"_"+str(model_type)

if model_type == "student_bi":
    LSTM_model.populate(mt_model)
else:
    W, b, embedding_parameters, LSTM_unit = dy.load(mt_model, LSTM_model)


''' forward pass '''
def forward_pass(x):
    input_sequence = [embedding_parameters[i] for i in x]
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
batch_size = 256    #MC uses 256
num_batches_training = int(np.ceil(len(train_words) / batch_size))
num_batches_testing = int(np.ceil(len(test_words) / batch_size))
####################


##### TESTING FUNCTION #####

mistakes = []
all_gold = []
all_preds = []

def test():
    all_predictions = []
    correct = 0
    incorrect = 0
    u_correct = 0
    u_incorrect = 0
    
    # make dicts of true/false pos/neg by pos tag
    TP = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    TN = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    FP = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    FN = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    TPu = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    TNu = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    FPu = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    FNu = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    
    # make dicts of precision/recall/F1 by pos tag
    Precision = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    Recall = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    F1 = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    Precision_unk = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    Recall_unk = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    F1_unk = dict(zip(list(tag_to_index(all_tags_short).values()),np.zeros(len(Counter(tags_collection)))))
    
    for batch in tqdm(range(num_batches_testing)):
        dy.renew_cg()
        # build the batch
        batch_words = test_words[batch*batch_size:(batch+1)*batch_size]
        batch_tags = test_tags[batch*batch_size:(batch+1)*batch_size]
        # iterate through the batch
        for sentence in range(len(batch_words)):
            # prepare input: words to indexes
            seq_of_idxs = word_to_index(batch_words[sentence], word_index)
            seq_of_tags = batch_tags[sentence]
            # make a forward pass
            preds = forward_pass(seq_of_idxs)
            tag_preds = predict(preds)
            all_predictions.append(tag_preds)
            for i in range(len(batch_words[sentence])):
                temp = []
                temp.extend((batch_words[sentence][i],batch_tags[sentence][i],tag_preds[i]))
                y = temp[1]    # the gold label
                y_hat = temp[2]    # the predicted label
                all_gold.append(index_tags[y])
                all_preds.append(index_tags[y_hat])
                if y == y_hat:
                    correct += 1
                    TP[y] = TP[y]+1
                    for key in TN:
                        if key != y:
                            TN[key] = TN[key] + 1
                elif y != y_hat:
                    mistakes.append(index_tags[y]+" "+index_tags[y_hat])
                    incorrect += 1
                    FP[y_hat] = FP[y_hat] +1
                    FN[y] = FN[y] + 1
                    for key in TN:
                        if key != y and key != y_hat:
                            TN[key] = TN[key] + 1
                if temp[0] not in known_words:
                    if y == y_hat:
                        u_correct += 1
                        TPu[y] = TPu[y]+1
                        for key in TNu:
                            if key != y:
                                TNu[key] = TNu[key] + 1
                    elif y != y_hat:
                        mistakes.append(index_tags[y]+" "+index_tags[y_hat])
                        u_incorrect += 1
                        FPu[y_hat] = FPu[y_hat] +1
                        FNu[y] = FNu[y] + 1
                        for key in TNu:
                            if key != y and key != y_hat:
                                TNu[key] = TNu[key] + 1

    overall_accuracy = float(correct)/float(correct+incorrect)
    unknown_accuracy = float(u_correct)/float(u_correct+u_incorrect)

    for key in Precision:
        if float(TP[key]+FP[key]) > 0:
            Precision[key] = float(TP[key])/float(TP[key]+FP[key])
        else:
            Precision[key] = 0
    for key in Recall:
        if float(TP[key]+FN[key]) > 0:
            Recall[key] = float(TP[key])/float(TP[key]+FN[key])
        else:
            Recall[key] = 0
    for key in F1:
        if float(2*TP[key]+FP[key]+FN[key]) > 0:
            F1[key] = float(2*TP[key])/float(2*TP[key]+FP[key]+FN[key])
        else:
            F1[key] = 0
            
    for key in Precision_unk:
        if float(TPu[key]+FPu[key]) > 0:
            Precision_unk[key] = float(TPu[key])/float(TPu[key]+FPu[key])
        else:
            Precision_unk[key] = 0
    for key in Recall_unk:
        if float(TPu[key]+FNu[key]) > 0:
            Recall_unk[key] = float(TPu[key])/float(TPu[key]+FNu[key])
        else:
            Recall_unk[key] = 0
    for key in F1_unk:
        if float(2*TPu[key]+FPu[key]+FNu[key]) > 0:
            F1_unk[key] = float(2*TPu[key])/float(2*TPu[key]+FPu[key]+FNu[key])
        else:
            F1_unk[key] = 0
    
    P_macro = float(sum(Precision.values()))/float(len(Precision.keys()))
    P_unk_macro = float(sum(Precision_unk.values()))/float(len(Precision_unk.keys()))
    
    R_macro = float(sum(Recall.values()))/float(len(Recall.keys()))
    R_unk_macro = float(sum(Recall_unk.values()))/float(len(Recall_unk.keys()))

    F1_macro = float(sum(F1.values()))/float(len(F1.keys()))
    F1_unk_macro = float(sum(F1_unk.values()))/float(len(F1_unk.keys()))

    print("Model: {}".format(mt_model))    
    print("Overall\n-------")
    print("Accuracy: {}".format(overall_accuracy))   
    print("Precision (macro): {}".format(P_macro))
    print("Recall (macro): {}".format(R_macro))
    print("F1 (macro): {}".format(F1_macro))
    
    print("\nUnknown Words\n-------------")
    print("Accuracy: {}".format(unknown_accuracy))
    print("Precision (macro): {}".format(P_unk_macro))
    print("Recall (macro): {}".format(R_unk_macro))
    print("F1 (macro): {}".format(F1_unk_macro))
    
    return overall_accuracy, P_macro, R_macro, F1_macro, unknown_accuracy, P_unk_macro, R_unk_macro, F1_unk_macro


##### USE THIS TO TEST #####
final_predictions = test()
#print(mistakes)



'''
for making confusion matrices
works well in jupyter notebooks
not so well via terminal
'''



# SMALL_SIZE = 24
# MEDIUM_SIZE = 32
# BIGGER_SIZE = 48

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# # In[10]:


# class_names = []
# for item in Counter(all_gold).keys():
#     class_names.append(item)
# class_names = sorted(class_names)
# print(class_names)


# # In[11]:


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()


# # Compute confusion matrix
# cnf_matrix = confusion_matrix(all_gold, all_preds)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure(figsize=(25,25))
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure(figsize=(25,25))
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()
