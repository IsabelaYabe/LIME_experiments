import numpy as np
import scipy as sp
from scipy import sparse
import argparse
import evaluate_explanations
import sys
import xgboost
sys.path.append('..')
from sklearn import ensemble
from sklearn import neighbors
import embedding_forest
def get_classifier(name, vectorizer):
  if name == 'logreg':
    return linear_model.LogisticRegression(fit_intercept=True)
  if name == 'random_forest':
    return ensemble.RandomForestClassifier(n_estimators=1000, random_state=1, max_depth=5, n_jobs=10)
  if name == 'svm':
    return svm.SVC(probability=True, kernel='rbf', C=10,gamma=0.001)
  if name == 'tree':
    return tree.DecisionTreeClassifier(random_state=1)
  if name == 'neighbors':
    return neighbors.KNeighborsClassifier()
  if name == 'embforest':
    return embedding_forest.EmbeddingForest(vectorizer)
class ParzenWindowClassifier:

    def __init__(self):
        # Definir o kernel como um método de instância ao invés de um lambda para flexibilidade
        self.kernel = self.kernel
    def kernel(self, x, sigma):
      if sp.sparse.issparse(x):
        # Para matrizes esparsas, use o método .multiply() e mantenha a esparsidade
        squared_sum = x.multiply(x).sum(axis=1)
      else:
        # Para matrizes densas, use np.power() ou operações equivalentes
        squared_sum = np.sum(np.power(x, 2), axis=1)
      return np.exp(-0.5 * squared_sum / (sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)

    def fit(self, X, y):
        self.X = X.toarray()
        self.y = y
        self.ones = y==1
        self.zeros = y==0
    def predict(self, x):
        b = sp.sparse.csr_matrix(x - self.X)
        pr = self.kernel(b, self.sigma)
        prob = sum(pr[self.ones]) / sum(pr)
        #print prob
        return int(prob > .5)
    def predict_proba(self, x):
        print('x-shape', x.shape)
        print('x type', type(x))
        x = x.toarray()     
        b = sp.sparse.csr_matrix(x - self.X)
        pr = self.kernel(b, self.sigma)
        prob = sum(pr[self.ones]) / sum(pr)
        return np.array([1 - prob, prob])
    def find_sigma(self, sigmas_to_try, cv_X, cv_y):
        self.sigma = sigmas_to_try[0]
        best_mistakes = 2**32 - 1
        best_sigma = self.sigma
        for sigma in sorted(sigmas_to_try):
            self.sigma = sigma
            preds = []
            for i in range(cv_X.shape[0]):
              preds.append(self.predict(cv_X[i]))
            mistakes = sum(cv_y != np.array(preds))
            print (sigma, mistakes)
            sys.stdout.flush()
            if mistakes < best_mistakes:
                best_mistakes = mistakes
                best_sigma = sigma
        print ('Best sigma achieves ', best_mistakes, 'mistakes. Disagreement=', float(best_mistakes) / cv_X.shape[0])
        self.sigma = best_sigma
    def explain_instance(self, x, _, __,num_features,___=None):
        x = x.toarray()
        print('x-shape', x.shape)
        print('x type', type(x))
        minus = self.X - x
        print("Foi")
        print('minus-shape', minus.shape)
        print('minus type', type(minus))
        print('minus', minus)
        b = sp.sparse.csr_matrix(minus)
        print("tratando segundo erro")
        ker = np.array([self.kernel(z.toarray().reshape(1, -1), self.sigma) for z in b])
        print('ker-shape', ker.shape)
        print('ker type', type(ker))
        print('ker', ker)
        print("verificação")
        
        ker = ker.reshape(-1, 1) 
        times = minus * ker
        
        sumk_0= sum(ker[self.zeros])
        sumk_1= sum(ker[self.ones])
        sumt_0 = sum(times[self.zeros])
        sumt_1 = sum(times[self.ones])
        sumk_total = sumk_0 + sumk_1
        exp = (sumk_0 * sumt_1 - sumk_1 * sumt_0) / (self.sigma **2 * sumk_total ** 2)
        print('exp', exp)
        features = x.nonzero()[1]
        print("eita")
 
        values = exp[x.nonzero()[1]]
        return sorted(zip(features, values), key=lambda x:np.abs(x[1]), reverse=True)[:num_features]
def main():
  parser = argparse.ArgumentParser(description='Visualize some stuff')
  parser.add_argument('--dataset', '-d', type=str, required=True,help='dataset name')
  parser.add_argument('--algorithm', '-a', type=str, required=True, help='algorithm_name')
  args = parser.parse_args()

  train_data, train_labels, test_data, test_labels, _ = LoadDataset(args.dataset)
  vectorizer = CountVectorizer(lowercase=False, binary=True)
  train_vectors = vectorizer.fit_transform(train_data)
  num_train = int(train_vectors.shape[0] * .8)
  indices = np.random.choice(range(train_vectors.shape[0]), train_vectors.shape[0], replace=False)
  train_v = train_vectors[indices[:num_train]]
  y_v = train_labels[indices[:num_train]]
  train_cv = train_vectors[indices[num_train:]]
  y_cv = train_labels[indices[num_train:]]
  print ('train_size', train_v.shape[0])
  print ('cv_size', train_cv.shape[0])
  classifier = get_classifier(args.algorithm, vectorizer)
  classifier.fit(train_v, y_v)
  print ('train accuracy:')
  print (accuracy_score(y_v, classifier.predict(train_v)))
  print ('cv accuracy:')
  print (accuracy_score(y_cv, classifier.predict(train_cv)))
  yhat_v = classifier.predict(train_v)
  yhat_cv = classifier.predict(train_cv)
  p = ParzenWindowClassifier()
  p.fit(train_v, yhat_v)
  p.find_sigma([0.1, .25, .5, .75, 1,2,3,4,5,6,7,8,9,10], train_cv, yhat_cv)
  print ('Best sigma:')
  print (p.sigma)

if __name__ == "__main__":
    main()
