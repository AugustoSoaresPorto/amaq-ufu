import numpy as np
import matplotlib.pyplot as plt
from dados import pontos

def main():
    # Dados de entrada
    entrada = np.array(pontos)
    x = entrada[:, :1]         # Extraindo a primeira coluna para x
    t = entrada[:, 1:]         # Extraindo a segunda coluna para t

    # Parâmetros
    neuronios_entrada = len(x[0])
    neuronios_escondidos = 10
    neuronios_saida = 1
    alpha = 0.05
    erro_total_admissivel = 0.001
    epocas = 10000
    
    # Treinamento da rede
    v, bv, w, bw, erro_quadratico_total, epoca, erro_total = treinar_rede_multicamadas(x, t, alpha, epocas, erro_total_admissivel, neuronios_entrada, neuronios_escondidos, neuronios_saida)

    print('-- DADOS --')
    print(f"Época {epoca}: Erro total = {erro_total[0]}")
    print()


    # Teste da rede treinada
    todos_y = []
    print('-- TESTE --')
    for padroes in range(len(x)):
        z_in = np.dot(x[padroes], v) + bv
        z = bipolar_sigmoid(z_in)
        
        y_in = np.dot(z, w) + bw
        y = bipolar_sigmoid(y_in)
        todos_y.append(y[0])
        
        print(f"Ponto {padroes+1}: t = {t[padroes][0]}   y = {y[0]}")

    # Plotagem do gráfico da função aproximada
    x_novo = np.linspace(0, 1, 100)  # Gera 100 pontos igualmente espaçados entre 0 e 1
    y_novo = []

    for x_i in x_novo:
        z_in = np.dot(np.array([[x_i]]), v) + bv
        z = bipolar_sigmoid(z_in)
        y_in = np.dot(z, w) + bw
        y = bipolar_sigmoid(y_in)
        y_novo.append(y[0])


    # Plotagem dos gráficos
    plt.figure(figsize=(13, 7))
    
    # Gráfico de comparação entre dados e função aproximada
    plt.subplot(1, 2, 1)
    plt.plot(x, t, 'bo', label='Dados Originais')
    plt.plot(x_novo, y_novo, 'r-', label='Função Aproximada')
    plt.title('Comparação entre Dados e Função Aproximada')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    # Gráfico do erro quadrático total
    plt.subplot(1, 2, 2)
    plt.plot(erro_quadratico_total, '.', color='green')
    plt.title('Curva do Erro Quadrático Total')
    plt.xlabel('Épocas')
    plt.ylabel('Erro quadrático')
    plt.grid(True)

    plt.show()

def treinar_rede_multicamadas(x, t, alpha, epocas, erro_total_admissivel, neuronios_entrada, neuronios_escondidos, neuronios_saida):
    # Inicialização dos pesos
    v = np.random.rand(neuronios_entrada, neuronios_escondidos) - 0.5
    bv = np.random.rand(neuronios_escondidos) - 0.5
    w = np.random.rand(neuronios_escondidos, neuronios_saida) - 0.5
    bw = np.random.rand() - 0.5

    # Treinamento da rede
    erro_quadratico_total = np.zeros(epocas)
    epoca = 0
    erro_total = 10

    for epoca in range(epocas):
        epoca += 1
        erro_total = 0
        
        for padroes in range(10):
            #  Fase de Feedforward
            z_in = np.dot(x[padroes], v) + bv
            z = bipolar_sigmoid(z_in)
            
            y_in = np.dot(z, w) + bw
            y = bipolar_sigmoid(y_in)
            
            #  Retropropagação do erro 
            d_k = (t[padroes] - y) * bipolar_sigmoid_derivative(y)
            D_w = alpha * d_k * z
            D_bw = alpha * d_k
            
            d_v = d_k * w.flatten() * bipolar_sigmoid_derivative(z)
            D_v = alpha * np.outer(x[padroes], d_v)
            D_bv = alpha * d_v
            
            # Atualização dos pesos
            w += D_w.reshape(-1, 1)
            bw += D_bw
            v += D_v
            bv += D_bv
            
            erro_total += 0.5 * ((t[padroes] - y) ** 2)
        
        erro_quadratico_total[epoca - 1] = erro_total[0]
        
        if erro_total < erro_total_admissivel:
            break
        
    return v, bv, w, bw, erro_quadratico_total, epoca, erro_total

# Função de ativação bipolar sigmoid
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Derivada da função de ativação bipolar sigmoid
def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

# Chamada da função principal
if __name__ == "__main__":
    main()