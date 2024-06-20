
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
    def __init__(self, sigma=25**2, num_samples=15000, K=6, alpha=0.1**(16), p=16,vectorizer=None, model=None, seed=42, generate="opt"):
        self.sigma = sigma # Parâmetro do kernel exponencial
        self.num_samples = num_samples # Número de amostras geradas 
        self.K = K # Número de palavras importantes
        self.alpha = alpha # Parâmetro de regularização Lasso
        self.model = model # Modelo de classificação
        self.vectorizer = vectorizer # Vetorizador
        self.generate = generate # Tipo de amostragem
        self.p = p # Precisão de impressão
        np.random.seed(seed) # Semente aleatória
        np.set_printoptions(precision=self.p) # Precisão de impressão
        

    # Binarizar vetor de palavras
    def binarize(self, x): 
        x = (x.todense() > 0).astype(int) # Binariza o vetor
        return x.getA1() # Retorna o vetor em formato de array

    def cossine_similarity(self, x, z):
        x = x.todense() # Transforma o vetor em uma matriz densa
        z = z.todense()
        dot_product = np.dot(x, z.T)
        norm_x = np.linalg.norm(x)
        norm_z = np.linalg.norm(z)
        return (dot_product / (norm_x * norm_z)).item() # Similaridade de cosseno
    
    # Define a função de kernel 
    def kernel(self, x, z):
        distance = self.cossine_similarity(x, z) # Similaridade de cosseno
        weight = np.exp((-(distance**2) / (self.sigma**2)))  # Kernel exponencial
        return weight
    
    
    def samples_simples(self, x):
        n = len(self.vectorizer.get_feature_names_out()) # Número de palavras no vetorizador
        x_indices = x.indices # Índices das palavras na sentença
        n_x_words = len(x.indices) # Número de palavras na sentença   
        sample_set = [np.zeros(n) for i in range(self.num_samples-1)] # Conjunto de amostras
        sample_set.append(self.binarize(x)) # Adiciona a sentença original binarizada
        Z_line_indices = [] # Índices das palavras ativadas
        for i in range(self.num_samples-1): # Gerar amostras aleatórias
            num_words = np.random.randint(1, n_x_words)  # Número de palavras na amostra
            # Escolhe aleatoriamente as palavras da sentença original
            # size = número de palavras na amostra
            # replace = False, não permite repetição
            z_line_indices = np.random.choice(x_indices, size=num_words, replace=False)
            # Ativa as palavras escolhidas
            sample_set[i][z_line_indices] = 1
            Z_line_indices.append(z_line_indices)
        Z_line_indices.append(x_indices)
        return sample_set, Z_line_indices

    # Gera dados ao redor de x_line relevancia das palavras
    def samples_opt(self, x):
        n = len(self.vectorizer.get_feature_names_out()) # Número de palavras no vetorizador
        x_indices = x.indices # Índices das palavras na sentença
        n_x_words = len(x.indices) # Número de palavras na sentença   
        sample_set = [np.zeros(n) for i in range(self.num_samples-1)] # Conjunto de amostras
        sample_set.append(self.binarize(x)) # Adiciona a sentença original binarizada
        weights = x.tocoo().data # Pesos das palavras
        weights_normalized = weights / weights.sum() # Normaliza os pesos
        Z_line_indices = [] # Índices das palavras ativadas
        for i in range(self.num_samples-1): # Gerar amostras aleatórias
            num_words = np.random.randint(1, n_x_words)  # Número de palavras na amostra
            # Escolhe aleatoriamente as palavras da sentença original
            # size = número de palavras na amostra
            # p = pesos normalizados, a probabilidade de escolher cada palavra
            # replace = False, não permite repetição
            z_line_indices = np.random.choice(x_indices, size=num_words, p=weights_normalized, replace=False)
            # Ativa as palavras escolhidas
            sample_set[i][z_line_indices] = 1
            Z_line_indices.append(z_line_indices)
        Z_line_indices.append(x_indices)
        return sample_set, Z_line_indices
    
    
    # Transforma um vetor em uma frase
    def sentences_samples(self, Z_line_indices):
        Z = []
        for z_line_indice in Z_line_indices:
            z=" ".join(self.vectorizer.get_feature_names_out()[z_line_indice])
            Z.append(self.vectorizer.transform([z]))
        return Z 

    # Define o vetor de pesos
    def LIME(self, x):
        if self.generate == "opt":
            Z_line, Z_line_indices = self.samples_opt(x)
        else:
            Z_line, Z_line_indices = self.samples_simples(x)
        Z=self.sentences_samples(Z_line_indices)
        Z_pred = np.array([self.model.predict(z)[0] for z in Z])  
        pi_x = np.array([self.kernel(x, z) for z in Z]) 
        lasso = Lasso(alpha=self.alpha)
        lasso.fit(Z_line, Z_pred, sample_weight=pi_x)
        w = lasso.coef_
        return w 
    

    # Gerar explicação
    def explain_instance(self, x):
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
      


# Lê textos
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.neural_network
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



# Exemplo de conjunto de dados de resenhas
dados = pd.read_csv('dados.csv', sep=';')

# Separar as features e o alvo
X = dados['review']
y = dados['sentimentos']

# Escolher uma instância para fazer a previsão
instance_index = 15
instance = X.iloc[instance_index]
instance_label = y.iloc[instance_index]
print('Instância:', instance)
print('Rótulo:', instance_label)

# Retira stopwords
stop_words = stopwords.words('portuguese')
X_sw = X.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# Vetorização do texto
vectorizer_sw = TfidfVectorizer()
X_vectorized_sw = vectorizer_sw.fit_transform(X_sw)
x_sw = X_vectorized_sw[instance_index]
# Dividir os dados em conjuntos de treino e teste
X_train_sw, X_test_sw, y_train_sw, y_test_sw = train_test_split(X_vectorized_sw, y, test_size=0.3, random_state=42)

# Treinamento do modelo de floresta aleatória sem stopwords
rf_model_sw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_sw.fit(X_train_sw, y_train_sw)

# Fazer previsões com o modelo de floresta aleatória sem stopwords
rf_y_pred_sw = rf_model_sw.predict(X_test_sw)

# Avaliar o modelo de floresta aleatória sem stopwords
print(f'Random Forest Accuracy (sem stopwords): {accuracy_score(y_test_sw, rf_y_pred_sw)}')
print(classification_report(y_test_sw, rf_y_pred_sw))

print("RANDOM FOREST")

print("Generator: opt")
LIME = LimeExplainerSentences(vectorizer=vectorizer_sw, model=rf_model_sw, generate="opt")
LIME.explain_instance(x_sw)

print("Generator: simples")
LIME = LimeExplainerSentences(vectorizer=vectorizer_sw, model=rf_model_sw, generate="simples")
LIME.explain_instance(x_sw)

print("#########################################")

print("MLP")

# Treinamento do modelo MLP sem stopwords
mlp_model_sw = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_model_sw.fit(X_train_sw, y_train_sw)

# Fazer previsões com o modelo MLP sem stopwords
mlp_y_pred_sw = mlp_model_sw.predict(X_test_sw)

# Avaliar o modelo MLP sem stopwords
print(f'MLP Accuracy (sem stopwords): {accuracy_score(y_test_sw, mlp_y_pred_sw)}')
print(classification_report(y_test_sw, mlp_y_pred_sw))

print("Generator: opt")
LIME = LimeExplainerSentences(vectorizer=vectorizer_sw, model=mlp_model_sw, generate="opt")
LIME.explain_instance(x_sw)

print("Generator: simples")
LIME = LimeExplainerSentences(vectorizer=vectorizer_sw, model=mlp_model_sw, generate="simples")
LIME.explain_instance(x_sw)

print("#########################################")

print("Gradient Boosting")

# Treinamento do modelo Gradient Boosting sem stopwords
gb_model_sw = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model_sw.fit(X_train_sw, y_train_sw)

# Fazer previsões com o modelo Gradient Boosting sem stopwords
gb_y_pred_sw = gb_model_sw.predict(X_test_sw)

# Avaliar o modelo Gradient Boosting sem stopwords
print(f'Gradient Boosting Accuracy (sem stopwords): {accuracy_score(y_test_sw, gb_y_pred_sw)}')

print(classification_report(y_test_sw, gb_y_pred_sw))

print("Generator: opt")

LIME = LimeExplainerSentences(vectorizer=vectorizer_sw, model=gb_model_sw, generate="opt")
LIME.explain_instance(x_sw)

print("Generator: simples")
LIME = LimeExplainerSentences(vectorizer=vectorizer_sw, model=gb_model_sw, generate="simples")
LIME.explain_instance(x_sw)

print("#########################################")



