import lime_functions 
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import cosine_similarity

class SubmodularPick:
    def __init__(self,X_n_vec, X, B=10,lime=None, vectorizer=None):
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