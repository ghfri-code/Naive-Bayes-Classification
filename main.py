import numpy as np
from preprocess import *
from Naivebayes import *


#Splitting into Testing and Training
X_train_amazon= np.array(X_amazon_clean[0:400] + X_amazon_clean[500:900])
Y_train_amazon = np.array(Y_amazon[0:400] + Y_amazon[500:900]).astype("int")
X_test_amazon = np.array(X_amazon_clean[400:500] + X_amazon_clean[900:1000])
Y_test_amazon = np.array(Y_amazon[400:500] + Y_amazon[900:1000]).astype('int')

X_train_yelp =  X_yelp_clean[0:400] + X_yelp_clean[500:900]
Y_train_yelp =  np.array(Y_yelp[0:400] + Y_yelp[500:900]).astype('int')
X_test_yelp =  X_yelp_clean[400:500] + X_yelp_clean[900:1000]
Y_test_yelp = np.array(Y_yelp[400:500] + Y_yelp[900:1000]).astype('int')

X_train_imdb =  X_imdb_clean[0:400] + X_imdb_clean[500:900]
Y_train_imdb =  np.array(Y_imdb[0:400] + Y_imdb[500:900]).astype('int')
X_test_imdb = X_imdb_clean[400:500] + X_imdb_clean[900:1000]
Y_test_imdb = np.array(Y_imdb[400:500] + Y_imdb[900:1000]).astype('int')

#extract unique words from each dataset
def unique(X):

    unique_words= set()
    for each_sentence in X:
        each_sentence = each_sentence.split(' ')
        for each_word in each_sentence:
            if (len(each_word) > 1):
                unique_words.add(each_word)
    return unique_words

X_amazon_unique = unique(X_train_amazon)
X_yelp_unique = unique(X_train_yelp)
X_imdb_unique = unique(X_train_imdb)


#Total & per-class accuracies (train & test for each website review separately)

print("Amazon\n")
nb = NaiveBayes(X_amazon_unique)
nb.fit(nb.calculate_feature_vector(X_train_amazon), Y_train_amazon)

print("Train Phase\n")
preds_train_amazon = nb.predict(nb.calculate_feature_vector(X_train_amazon))
preds0_train_amazon , preds1_train_amazon = nb.accuracy_per_class(preds_train_amazon ,Y_train_amazon)
print(f'Total Accuracy: {(preds_train_amazon == Y_train_amazon).mean()*100}')
print("Predicted correctly {} out of {} from class 0 ({:.2f}%)".format(preds0_train_amazon,len(Y_train_amazon),(preds0_train_amazon/len(Y_train_amazon))*100))
print("Predicted correctly {} out of {} from class 1 ({:.2f}%)".format(preds1_train_amazon,len(Y_train_amazon),(preds1_train_amazon/len(Y_train_amazon))*100))

print("\nTest Phase\n")
preds_test_amazon = nb.predict(nb.calculate_feature_vector(X_test_amazon))
preds0_test_amazon , preds1_test_amazon = nb.accuracy_per_class(preds_test_amazon ,Y_test_amazon)
print(f'Total Accuracy: {(preds_test_amazon == Y_test_amazon).mean()*100}')
print("Predicted correctly {} out of {} from class 0 ({:.2f}%)".format(preds0_test_amazon,len(Y_test_amazon),(preds0_test_amazon/len(Y_test_amazon))*100))
print("Predicted correctly {} out of {} from class 1 ({:.2f}%)".format(preds1_test_amazon,len(Y_test_amazon),(preds1_test_amazon/len(Y_test_amazon))*100))



print("\nYelp\n")
nb = NaiveBayes(X_yelp_unique)
nb.fit(nb.calculate_feature_vector(X_train_yelp), Y_train_yelp)

print("Train Phase\n")
preds_train_yelp = nb.predict(nb.calculate_feature_vector(X_train_yelp))
preds0_train_yelp , preds1_train_yelp = nb.accuracy_per_class(preds_train_yelp ,Y_train_yelp)
print(f'Total Accuracy: {(preds_train_yelp == Y_train_yelp).mean()*100}')
print("Predicted correctly {} out of {} from class 0 ({:.2f}%)".format(preds0_train_yelp,len(Y_train_yelp),(preds0_train_yelp/len(Y_train_yelp))*100))
print("Predicted correctly {} out of {} from class 1 ({:.2f}%)".format(preds1_train_yelp,len(Y_train_yelp),(preds1_train_yelp/len(Y_train_yelp))*100))

print("\nTest Phase\n")
preds_test_yelp = nb.predict(nb.calculate_feature_vector(X_test_yelp))
preds0_test_yelp , preds1_test_yelp = nb.accuracy_per_class(preds_test_yelp ,Y_test_yelp)
print(f'Total Accuracy: {(preds_test_yelp == Y_test_yelp).mean()*100}')
print("Predicted correctly {} out of {} from class 0 ({:.2f}%)".format(preds0_test_yelp,len(Y_test_yelp),(preds0_test_yelp/len(Y_test_yelp))*100))
print("Predicted correctly {} out of {} from class 1 ({:.2f}%)".format(preds1_test_yelp,len(Y_test_yelp),(preds1_test_yelp/len(Y_test_yelp))*100))


print("\nIMDB\n")
nb = NaiveBayes(X_imdb_unique)
nb.fit(nb.calculate_feature_vector(X_train_imdb), Y_train_imdb)

print("Train Phase\n")
preds_train_imdb = nb.predict(nb.calculate_feature_vector(X_train_imdb))
preds0_train_imdb , preds1_train_imdb = nb.accuracy_per_class(preds_train_imdb ,Y_train_imdb)
print(f'Total Accuracy: {(preds_train_imdb == Y_train_imdb).mean()*100}')
print("Predicted correctly {} out of {} from class 0 ({:.2f}%)".format(preds0_train_imdb,len(Y_train_imdb),(preds0_train_imdb/len(Y_train_imdb))*100))
print("Predicted correctly {} out of {} from class 1 ({:.2f}%)".format(preds1_train_imdb,len(Y_train_imdb),(preds1_train_imdb/len(Y_train_imdb))*100))

print("\nTest Phase\n")
preds_test_imdb = nb.predict(nb.calculate_feature_vector(X_test_imdb))
preds0_test_imdb , preds1_test_imdb = nb.accuracy_per_class(preds_test_imdb ,Y_test_imdb)
print(f'Total Accuracy: {(preds_test_imdb == Y_test_imdb).mean()*100}')
print("Predicted correctly {} out of {} from class 0 ({:.2f}%)".format(preds0_test_imdb,len(Y_test_imdb),(preds0_test_imdb/len(Y_test_imdb))*100))
print("Predicted correctly {} out of {} from class 1 ({:.2f}%)".format(preds1_test_imdb,len(Y_test_imdb),(preds1_test_imdb/len(Y_test_imdb))*100))

