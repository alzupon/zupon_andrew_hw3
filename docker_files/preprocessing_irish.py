import pickle
import sys

##### IMPORT DATA FILES #####
'''Import training file'''
collection = open(sys.argv[1], 'r')
training_data = collection.read()
collection.close()

'''Prepare training data into a list of sentences'''
training_data = training_data.strip()
train_data = training_data.split('\n')
train_data2 = []
for item in train_data:
    item = item[8:].lstrip()
    item = item.split('  ')
    train_data2.append(item)
train_data3 = []
for item in train_data2:
    temp = []
    for x in item:
        x = x.lower()
        y = x.split(" /")
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
kw_irish = []
for sentence in train_words:
    for word in sentence:
        word2 = word.lower()
        kw_irish.append(word2)

count = 0
for senttence in train_words:
    for item in sentence:
        count += 1

print("Number of Irish words:", count)


with open('irish_words_5000.txt', 'wb') as irish_w:
    pickle.dump(train_words, irish_w)
with open('irish_tags_5000.txt', 'wb') as irish_t:
    pickle.dump(train_tags, irish_t)

with open('known_words_irish_5000.txt', 'wb') as known_w:
    pickle.dump(kw_irish, known_w)