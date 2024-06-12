
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import cosine_similarity

class LimeExplainer:
    def __init__(self, kernel_width=0.2):
        self.kernel_width = kernel_width
    
    # Define a função de kernel e a distância entre xi e X_lime
    def kernel_fn(self, xi, X_lime):
        distances = np.sum((xi - X_lime)**2, axis=1)  # Distância euclidiana
        weights = np.sqrt(np.exp(-(distances**2) / (self.kernel_width**2)))  # Kernel gaussiano
        return weights
    
    # Gera os dados ao redor de xi
    def generate_data(self, xi, num_samples, num_features):
        # Cria os dados com o mesmo número de features que xi
        X_lime = np.random.normal(0, 1, size=(num_samples,xi.shape[0]))
        # Zera alguns valores e deixa apenas num_features valores diferentes de zero
        for i in range(num_samples):
            zero_indices = np.random.choice(xi.shape[0], xi.shape[0] - num_features, replace=False)
            X_lime[i, zero_indices] = 0
        return X_lime

    # Explica uma instância xi
    def explain_instance(self, xi, num_samples, num_features, model):
        # Gera os dados ao redor de xi
        X_lime = self.generate_data(xi, num_samples, num_features)
        # Calcula a predição para esses dados
        y_lime = model.predict(X_lime)
        # Calcula os pesos para cada dado
        weights = self.kernel_fn(xi, X_lime)
        # Ajusta um modelo linear simples
        simpler_model = LogisticRegression()
        simpler_model.fit(X_lime, y_lime, sample_weight=weights)
        return X_lime, y_lime,weights, simpler_model.coef_
    
class LimeExplainerSentences:
    def __init__(self, sigma=0.2, num_samples=1000, K=5, alpha=0.1, vectorizer=None, model=None):
        self.sigma = sigma
        self.num_samples = num_samples
        self.K = K
        self.alpha = alpha
        self.model = model
        self.vectorizer = vectorizer
    
    # Binarizar vetor de palavras
    def binarize(self, x):
        # 
        n = len(self.vectorizer.get_feature_names_out())
        x_bin=np.zeros(n, dtype=int)
        for i in x.indices:
            x_bin[i] = 1
        return x_bin

    # Define a função de kernel 
    def kernel(self, x, z):
        distance = cosine_similarity(x, z) # Similaridade de cosseno
        weights = np.sqrt(np.exp(-(distance**2) / (self.sigma**2)))  # Kernel exponencial
        return weights
        
    # Gera dados ao redor de x_line
    def samples(self, x):
        # Quantidade de palavras
        n = len(self.vectorizer.get_feature_names_out())
        x_indices = x.indices
        n_x_words = len(x.indices)    
        sample_set = [np.zeros(n) for i in range(self.num_samples-1)]
        sample_set.append(self.binarize(x))
        # Gera amostras
        for i in range(self.num_samples-1):
            z_line_indices = np.random.randint(2, size=n_x_words-1)
            z_line_indices = np.where(z_line_indices == 1)[0]
            activated_words = [x_indices[j] for j in z_line_indices]
            sample_set[i][activated_words] = 1
        return sample_set
    
    # Transforma um vetor em uma frase
    def sentences_samples(self, z_line):
        indices = np.where(z_line == 1)[0]
        z=" ".join([self.vectorizer.get_feature_names_out()[indice] for indice in indices])
        return self.vectorizer.transform([z])

    # Define o vetor de pesos
    def LIME(self, x):
        # Gera amostras
        Z_line = self.samples(x)
        Z=[]
        for i in range(len(Z_line)):
            Z.append(self.sentences_samples(Z_line[i]))
        Z_pred = [self.model.predict(z) for z in Z]
        pi_x = []
        for z in Z:
            pi_x.append(self.kernel(x, z)) 
        lasso = Lasso(alpha=self.alpha)
        lasso.fit(Z_line, Z_pred, sample_weight=pi_x)
        w = lasso.coef_
        return w

    # Gerar explicação
    def explain_instance(self, x):
        # Obtém pesos 
        w = self.LIME(x)
        # Obtém o módulo dos pesos
        abs_valores = np.abs(w)
        # Obtém os índices dos K pesos mais importantes
        indices = np.argsort(abs_valores)[::-1][:self.K]
        print(f"Frase: {x}")
        print(f"w: {w}")
        print("Palavras importantes:")
        for i, word in enumerate(x):
            if i in indices:
                print(f"{word}: {w[i]}")    
