
import numpy as np
import random
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
    
# Lime para explicação de texto    


class LimeExplainerSentences:
    def __init__(self, sigma=0.1, num_samples=15000, K=3, alpha=0.1**(16), p=16,vectorizer=None, model=None, seed=42):
        self.sigma = sigma
        self.num_samples = num_samples
        self.K = K
        self.alpha = alpha
        self.p = p
        self.model = model
        self.vectorizer = vectorizer
        np.random.seed(seed)
        random.seed(42)

    # Binarizar vetor de palavras
    def binarize(self, x):
        np.set_printoptions(precision=self.p)
        n = len(self.vectorizer.get_feature_names_out())
        x_bin=np.zeros(n, dtype=int)
        for i in x.indices:
            x_bin[i] = 1
        return x_bin

    # Define a função de kernel 
    def kernel(self, x, z):
        np.set_printoptions(precision=self.p)
        distance = cosine_similarity(x, z) # Similaridade de cosseno
        weights = np.sqrt(np.exp(-(distance**2) / (self.sigma**2)))  # Kernel 
        return weights
        
    # Gera dados ao redor de x_line
    def samples(self, x):
        np.set_printoptions(precision=self.p)
        n = len(self.vectorizer.get_feature_names_out())
        x_indices = x.indices
        n_x_words = len(x.indices)    
        sample_set = [np.zeros(n) for i in range(self.num_samples-1)]
        sample_set.append(self.binarize(x))
        for i in range(self.num_samples-1):
            z_line_indices = np.random.randint(2, size=n_x_words)
            while not np.any(z_line_indices):
                z_line_indices = np.random.randint(2, size=n_x_words)
            z_line_indices = np.where(z_line_indices == 1)[0]
            activated_words = [x_indices[j] for j in z_line_indices]
            sample_set[i][activated_words] = 1
        return sample_set
    
    # Transforma um vetor em uma frase
    def sentences_samples(self, z_line):
        np.set_printoptions(precision=self.p)
        indices = np.where(z_line == 1)[0]
        z=" ".join([self.vectorizer.get_feature_names_out()[indice] for indice in indices])
        return self.vectorizer.transform([z])

    # Define o vetor de pesos
    def LIME(self, x):
        np.set_printoptions(precision=self.p)
        Z_line = self.samples(x)
        Z=[]
        for i in range(len(Z_line)):
            Z.append(self.sentences_samples(Z_line[i]))
        Z_pred = np.array([self.model.predict(z)[0] for z in Z])  
        pi_x = np.array([self.kernel(x, z)[0][0] for z in Z]) 
        lasso = Lasso(alpha=self.alpha)
        lasso.fit(Z_line, Z_pred, sample_weight=pi_x)
        w = lasso.coef_
        return w 

    # Gerar explicação
    def explain_instance(self, x):
        np.set_printoptions(precision=self.p)
        w = self.LIME(x)
        abs_valores = np.abs(w)
        indices = np.argsort(abs_valores)[::-1][:self.K]
        print("Palavras importantes:")
        for i in indices:
            print(f"{self.vectorizer.get_feature_names_out()[i]}: {w[i]}")    


class SubmodularPick:
    def __init__(self,X_n_vec, X, B,lime=None, vectorizer=None):
        self.X_n_vec = X_n_vec
        self.X = X
        self.B = B
        self.lime = lime
        self.vectorizer = vectorizer

    def explain_instances(self, X):
        W = [self.lime.LIME(x) for x in X] 
        return np.array(W)      
        
    def importancia(self, W):
        I = np.zeros(W.shape[1])  
        for j in range(W.shape[1]):
            soma = np.sum(np.abs(W[:, j]))
            I[j] = np.sqrt(soma)
        return I

    def cobertura(self, V, W, I):
        c_value = 0  
        for j in range(W.shape[1]):
            if any(W[i, j] > 0 for i in V):  # Se a característica j é relevante
                c_value += I[j]
        return c_value

    def guloso(self, X, B):
        W = self.explain_instances(X) 
        I = self.importancia(W)
        nao_selecionados = list(range(W.shape[0]))  # Lista de índices não selecionados
        V = []  # Conjunto de características selecionadas
        itens = 0  # Número de elementos selecionados
        c_value = 0  # Valor de c

        while itens < B and nao_selecionados:
            best_gain = -np.inf  # Valor de ganho máximo
            best_item = None  # Índice do melhor item 

            for item in nao_selecionados:  # Itera sobre os itens não selecionados
                lista_temp = V + [item]
                gain = self.cobertura(lista_temp, W, I) - self.cobertura(V, W, I)
                if gain > best_gain:
                    best_gain = gain  # Atualiza o valor de ganho máximo
                    best_item = item  # Atualiza o melhor item

            if best_item is not None:
                V.append(best_item)  # Adiciona o melhor item ao conjunto
                nao_selecionados.remove(best_item)  # Remove o melhor item dos não selecionados
                itens += 1
                c_value += best_gain

        return V
    
    def explain_model(self, X):
        V = self.guloso(X, self.B)
        print("Melhor conjunto de explicação: ", V)
        for i in V:
            print(f"{self.X_n_vec[i]}")
        print("Fim da explicação")        
