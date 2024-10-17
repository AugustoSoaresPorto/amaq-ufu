import numpy as np
import matplotlib.pyplot as plt  # Nova importação para a visualização
from database_iris import dados

def main():
    flores_teste = [
        [5.1, 3.5, 1.4, 0.2],   # Setosa
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2], 
        [6.5, 2.8, 4.6, 1.5],   # Versicolor
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],
        [6.3, 3.3, 6.0, 2.5],   # Virginica
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.5, 3.0, 5.8, 2.2],
        [7.6, 3.0, 6.6, 2.1]
    ]

    # Normalização das entradas
    entrada = (dados - np.mean(dados, axis=0)) / np.std(dados, axis=0)
    saida = np.array([[1, 0, 0]] * 50 + [[0, 1, 0]] * 50 + [[0, 0, 1]] * 50)

    neuronios_entrada = 4
    neuronios_escondidos = 15  
    neuronios_saida = 3
    alpha = 0.1  
    erro_total_admissivel = 0.001
    epocas = 10001

    print(f" -- Treinamento da MLP com {neuronios_entrada} neurônios de entrada --")

    v, bv, w, bw = treinar_rede_multicamadas(
        entrada, saida, alpha, epocas, erro_total_admissivel, 
        neuronios_entrada, neuronios_escondidos, neuronios_saida
    )

    print("\n -- Teste da MLP --")

    flores_teste_normalizadas = (flores_teste - np.mean(dados, axis=0)) / np.std(dados, axis=0)
    resultados = testar_rede(flores_teste_normalizadas, v, bv, w, bw)
    
    casos = ["Setosa", "Versicolor", "Virginica"]
    precisao = 0
    for i, res in enumerate(resultados):
        if i < 5:
            if res == 0:
                precisao += 1
        elif i < 10 and i >= 5:
            if res == 1:
                precisao += 1
        else:
            if res == 2:
                precisao += 1
                
        print(f"Previsão para flor {i + 1}: Iris-{casos[res]}")
        
    print(f"\nPrecisão: {precisao / len(flores_teste) * 100}%")
    
    # Plotar o erro quadrático total
    plt.plot(erro_quadratico_total)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Total')
    plt.title('Curva do Erro Quadrático Total')
    plt.show()

def treinar_rede_multicamadas(x, t, alpha, epocas, erro_total_admissivel, neuronios_entrada, neuronios_escondidos, neuronios_saida):
    v = np.random.randn(neuronios_entrada, neuronios_escondidos) * 0.1
    bv = np.random.randn(neuronios_escondidos) * 0.1
    w = np.random.randn(neuronios_escondidos, neuronios_saida) * 0.1
    bw = np.random.randn(neuronios_saida) * 0.1

    global erro_quadratico_total  # Para visualização posterior
    erro_quadratico_total = np.zeros(epocas)

    for epoca in range(epocas):
        erro_total = 0

        for i in range(len(x)):
            z_in = np.dot(x[i], v) + bv
            z = sigmoid(z_in)

            y_in = np.dot(z, w) + bw
            y = sigmoid(y_in)

            d_k = (t[i] - y) * sigmoid_derivative(y)
            D_w = alpha * np.outer(z, d_k)
            D_bw = alpha * d_k

            d_v = np.dot(d_k, w.T) * sigmoid_derivative(z)
            D_v = alpha * np.outer(x[i], d_v)
            D_bv = alpha * d_v

            w += D_w
            bw += D_bw
            v += D_v
            bv += D_bv

            erro_total += 0.5 * np.sum((t[i] - y) ** 2)

        erro_quadratico_total[epoca] = erro_total

        if erro_total < erro_total_admissivel:
            print(f"Treinamento concluído na época {epoca + 1} com erro total {erro_total}.")
            break

        if epoca % 1000 == 0:
            print(f"Época {epoca + 1}: Erro = {erro_total}")

    return v, bv, w, bw

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Função de ativação bipolar sigmoid
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Derivada da função de ativação bipolar sigmoid
def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

def testar_rede(X_teste, v, bv, w, bw):
    z_in = np.dot(X_teste, v) + bv
    z = sigmoid(z_in)

    y_in = np.dot(z, w) + bw
    y = sigmoid(y_in)

    return np.argmax(y, axis=1)

if __name__ == "__main__":
    main()
