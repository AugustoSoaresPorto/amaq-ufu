from dicionario_5x5 import letras


def main():
    # Treinar o perceptron
    taxa_aprendizado = 0.05 # α adotado
    w, b = algoritmo_perceptron(letras, taxa_aprendizado)

    # Testar com uma entrada nova
    x = [-1, -1, 1, -1, -1,
            -1, 1, -1, -1, -1,
            1, -1, -1, -1, 1,
            1, -1, -1, 1, 1,
            1, -1, -1, -1, 1]

    y = teste_perceptron(x, w, b)

    # Exibir o resultado do perceptron
    estimativa = None

    for i, letra in enumerate(letras.keys()):
        print(f"Saída para a letra {letra}: y = {y[i]}")
        if y[i] == 1:
            estimativa = letras[letra]

    print("Entrada:")
    print_letra(x)

    if estimativa is None:
        print("Não foi possível identificar a letra")
    else:
        print("Saída estimada:")
        print_letra(estimativa)


# Função para imprimir a letra
def print_letra(letra):
    for i in range(5):
        for j in range(5):
            print('#' if letra[i*5 + j] == 1 else ' ', end='')
        print()


# Função de ativação
def ativacao(yliq): 
    if yliq >= 0:  # adotei θ = 0
        return 1
    else:
        return -1
    

# Função para treinar o perceptron
def algoritmo_perceptron(letras, taxa_aprendizado):
    num_letras = len(letras)
    w = [[0] * 25 for _ in range(num_letras)]
    b = [0] * num_letras
    alfa = taxa_aprendizado

    while True:
        pesos_mudaram = False
        
        for i, (letra, valores) in enumerate(letras.items()):
            x = valores
            
            t = []
            for j in range(num_letras):
                if j == i:
                    t.append(1)
                else:
                    t.append(-1)

            for j in range(num_letras):
                yliq = sum(w[j][k] * x[k] for k in range(25)) + b[j]
                y = ativacao(yliq)

                if y != t[j]:
                    pesos_mudaram = True
                    for k in range(25):
                        w[j][k] += alfa * x[k] * t[j]
                    b[j] += alfa * t[j]
        
        if not pesos_mudaram:
            break

    return w, b


# Função para testar o perceptron
def teste_perceptron(x, w, b):
    y = []
    for i in range(len(w)):
        yliq = sum(w[i][k] * x[k] for k in range(25)) + b[i]
        y.append(ativacao(yliq))
    return y


if __name__ == '__main__':
    main()
