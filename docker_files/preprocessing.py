import pickle
import sys

##### IMPORT DATA FILES #####
'''Import training file'''
#collection = open(sys.argv[1], 'r')
collection = open("train_all.txt", 'r')
training_data = collection.read()
training_data = training_data.strip()
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
    stemp = [["#","#"]]
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
collection = open("dev_all.txt", 'r')
#collection = open("bigtest_all.txt", 'r')
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
    stemp = [["#","#"]]
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
#collection = open("dev_all.txt", 'r')
collection = open("bigtest_all.txt", 'r')
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
    stemp = [["#","#"]]
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

with open('train_words.txt', 'wb') as train_w:
    pickle.dump(train_words, train_w)
with open('train_tags.txt', 'wb') as train_t:
    pickle.dump(train_tags, train_t)

with open('dev_words.txt', 'wb') as dev_w:
    pickle.dump(dev_words, dev_w)
with open('dev_tags.txt', 'wb') as dev_t:
    pickle.dump(dev_tags, dev_t)

with open('test_words.txt', 'wb') as test_w:
    pickle.dump(test_words, test_w)
with open('test_tags.txt', 'wb') as test_t:
    pickle.dump(test_tags, test_t)

with open('known_words.txt', 'wb') as known_w:
    pickle.dump(kw, known_w)

all_tags = train_tags+dev_tags+test_tags
