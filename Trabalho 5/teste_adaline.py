import random
import matplotlib.pyplot as pyplot

from base_de_dados import dados 

def main():
    # Separando entradas (entradas_x) e saídas (t)
    entradas_x = [dado[:-1] for dado in dados]  #s1 e s2
    t = [dado[-1] for dado in dados]   #t

    alpha=0.01
    tolerancia=0.001
    max_epochs=200

    # Treinamento
    w, b, erros_quadraticos = treinar_adaline(entradas_x, t,alpha, tolerancia, max_epochs)

    # Plotando o erro quadrático total por época
    pyplot.plot(erros_quadraticos)
    pyplot.title('Erro Quadrático Total durante o Treinamento')
    pyplot.xlabel('Épocas')
    pyplot.ylabel('Erro Quadrático Total')
    pyplot.show()

    # Testando a rede com os dados de treinamento
    y_pred = testar_adaline(entradas_x, w, b)

    # Exibindo resultados
    acuracia = 0
    for i, (xi, yi_real, yi_pred) in enumerate(zip(entradas_x, t, y_pred)):
        print(f"Amostra {i+1}: Real: {yi_real}, Previsto: {yi_pred}")
        if yi_real == yi_pred:
            acuracia += 1 
    print(f"Taxa de acurácia: {100*acuracia/len(y_pred)}%")

# Função de treinamento do Adaline
def treinar_adaline(entradas_x, t, alpha, tolerancia, max_epochs):
    random.seed(42)  # Para resultados reprodutíveis
    n_amostras = len(entradas_x)
    n_entradas = len(entradas_x[0])
    
    # Inicializar pesos aleatórios entre -0.5 e +0.5
    w = [random.uniform(-0.5, 0.5) for i in range(n_entradas)]
    b = random.uniform(-0.5, 0.5)
    
    erros_quadraticos = []  # Para armazenar o erro quadrático total em cada época

    for epoch in range(max_epochs):
        erro_total = 0
        maior_delta = 0
        
        for i in range(n_amostras):
            xi = entradas_x[i]  # Entradas (s1, s2)
            yi_liq = prod_escalar(w, xi) + b  # yl = Σ wi * xi + b
            erro = t[i] - yi_liq  # t - yliq
            
            # Atualização dos pesos e bias
            for j in range(n_entradas):
                delta_w = alpha * erro * xi[j]
                w[j] += delta_w
                maior_delta = max(maior_delta, abs(delta_w))
            delta_b = alpha * erro
            b += delta_b
            maior_delta = max(maior_delta, abs(delta_b))

            # Acumular erro quadrático
            erro_total += erro**2

        erros_quadraticos.append(erro_total)
        
        # Condição de parada: se maior alteração nos pesos for menor que a tolerância
        if maior_delta < tolerancia:
            print(f"Convergência atingida na época {epoch+1}")
            break
    
    return w, b, erros_quadraticos

# Função para testar a rede Adaline treinada
def testar_adaline(entradas_x, w, b):
    resultados = []
    for xi in entradas_x:
        yi_liq = prod_escalar(w, xi) + b
        y = 1 if yi_liq >= 0 else -1  # Limite para classificar como 1 ou -1
        resultados.append(y)
    return resultados

# Produto escalar
def prod_escalar(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

# Executar o programa
if __name__ == '__main__':
    main()
