import numpy as np

class NaiveBayes(object):

    def __init__(self, unique_words,alpha=1.0):

        self.alpha = alpha
        self.prior = None
        self.word_counts = None
        self.word_proba = None
        self.unique_words = unique_words

    #Fit training data for Naive Bayes classifier
    def fit(self, X, y):

        n = X.shape[0]

        X_by_class = np.array([X[y == c] for c in np.unique(y)])
        self.prior = np.array([len(X_class) / n for X_class in X_by_class])
        self.word_counts = np.array([sub_arr.sum(axis=0) for sub_arr in X_by_class]) + self.alpha
        self.lk_word = self.word_counts / (self.word_counts.sum(axis=1).reshape(-1, 1)+ 2)


        return self

    # Feature Vector for Training and Testing Data
    def calculate_feature_vector(self , X):
      feature_vectors=[]
      for i in X:
          #word_list=i.split(" ")
          feature_vector=[]
          for j in self.unique_words:
              feature_vector.append(i.count(j))
          feature_vectors.append(feature_vector)

      return np.array(feature_vectors)

    #Predict probability of class membership
    def predict_proba(self, X):


        # loop over each observation to calculate conditional probabilities
        class_numerators = np.zeros(shape=(X.shape[0], self.prior.shape[0]))
        for i, x in enumerate(X):
            word_exists = x.astype(bool)
            lk_words_present = self.lk_word[:, word_exists]
            lk_message =(lk_words_present).prod(axis=1)
            class_numerators[i] = lk_message * self.prior


        conditional_probas = class_numerators
        return conditional_probas

    #Predict class with highest probability
    def predict(self, X):

        return self.predict_proba(X).argmax(axis=1)


    def accuracy_per_class(self, preds, y):

      correct_predictions0 = 0
      correct_predictions1 = 0
      predictions_list0 = []
      predictions_list1 = []

      for i in range(len(y)):

        if y[i] == 0:
          if preds[i] == y[i]:
            correct_predictions0 += 1
            predictions_list0.append(1)
          else:
            predictions_list0.append(0)

        elif y[i] == 1:
            if preds[i] == y[i]:
              correct_predictions1 += 1
              predictions_list1.append(1)
            else:
              predictions_list1.append(0)

      return (correct_predictions0,correct_predictions1)
