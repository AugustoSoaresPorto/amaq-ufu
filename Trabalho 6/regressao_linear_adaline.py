import random
import matplotlib.pyplot as pyplot
import numpy as np
from base_de_dados import dados 

def main():
    entradas_x = [dado[0] for dado in dados]   # x
    entradas_y = [dado[1] for dado in dados]   # y

    alpha = 0.01
    tolerancia = 0.001
    max_epochs = 200
    
    # Treinamento
    w, b, erros_quadraticos = treinar_adaline(entradas_x, entradas_y, alpha, tolerancia, max_epochs)
    
    # Teste
    y_pred = testar_adaline(entradas_x, w, b)
    
    # Exibindo resultados
    for i in range(len(y_pred)):
        yi_real = entradas_y[i]
        yi_pred = y_pred[i]
        print(f"Amostra {i+1}: Real: {yi_real}, Previsto: {yi_pred}")
    
    # Traçar a linha de regressão linear obtida pela Adaline
    pyplot.scatter(entradas_x, entradas_y, color="cyan", label="Observações")
    pyplot.plot(entradas_x, y_pred, color="red", label="Regressão Adaline", linewidth=2)
    
    # Comparar com as equações de a e b tradicionais
    n = len(entradas_x)
    soma_x = sum(entradas_x)
    soma_y = sum(entradas_y)
    soma_xy = sum(x * y for x, y in zip(entradas_x, entradas_y))
    soma_x2 = sum(x ** 2 for x in entradas_x)
    
    coeficiente_angular = (n * soma_xy - soma_x * soma_y) / (n * soma_x2 - soma_x ** 2)
    coeficiente_linear = (soma_y - coeficiente_angular * soma_x) / n

    print(f"Equação de regressão tradicional: y = {coeficiente_angular} x + {coeficiente_linear}")
    print(f"Equação de regressão pela Adaline: y = {w} x + {b}")

    # Traçar a linha de regressão linear obtida pelas equações tradicionais
    y_trad = [coeficiente_angular * x + coeficiente_linear for x in entradas_x]
    pyplot.plot(entradas_x, y_trad, color="green", linestyle='dashed', label="Regressão Tradicional", linewidth=2)

    # Calcular coeficiente de correlação de Pearson
    media_x = soma_x / n
    media_y = soma_y / n
    
    r_num = sum((x - media_x) * (y - media_y) for x, y in zip(entradas_x, entradas_y))
    r_den_x = sum((x - media_x) ** 2 for x in entradas_x)
    r_den_y = sum((y - media_y) ** 2 for y in entradas_y)
    
    r = r_num / np.sqrt(r_den_x * r_den_y)
    print(f"Coeficiente de Correlação de Pearson (r): {r}")
    
    r= r ** 2
    print(f"Coeficiente de Determinação (r^2): {r}")
    
    # Plotando o gráfico final
    pyplot.xlabel("Eixo X")
    pyplot.ylabel("Eixo Y")
    pyplot.legend()
    pyplot.show()
    

# Função de treinamento do Adaline
def treinar_adaline(entradas_x, t, alpha, tolerancia, max_epochs):
    w = random.uniform(-0.5, 0.5)  # Inicialização do peso
    b = random.uniform(-0.5, 0.5)  # Inicialização do bias
    erros_quadraticos = []

    for epoch in range(max_epochs):
        erro_total = 0
        maior_delta = 0.0

        for i in range(len(entradas_x)):
            xi = entradas_x[i]
            yi_real = t[i]
            yi_pred = w * xi + b  # saída da Adaline

            erro = yi_real - yi_pred
            erro_total += erro ** 2

            # Atualização dos pesos e bias
            w += alpha * erro * xi
            b += alpha * erro
            
            maior_delta = max(maior_delta, abs(w))
            
        erro_quadratico_medio = erro_total / len(entradas_x)
        erros_quadraticos.append(erro_quadratico_medio)

        if erro_quadratico_medio < tolerancia:
            break

        if maior_delta < tolerancia:
            break

    return w, b, erros_quadraticos

# Função para testar a rede Adaline treinada
def testar_adaline(entradas_x, w, b):
    y_pred = []
    for xi in entradas_x:
        yi_pred = w * xi + b
        y_pred.append(yi_pred)
    return y_pred

# Executar o programa
if __name__ == "__main__":
    main()
