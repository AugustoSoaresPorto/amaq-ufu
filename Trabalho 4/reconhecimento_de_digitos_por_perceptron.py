from digitos_20x20 import digitos


def main():
    # Treinar o perceptron
    taxa_aprendizado = 0.020 # α adotado
    w, b = algoritmo_perceptron(digitos, taxa_aprendizado)

    # Testar com uma entrada nova
    x = [
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1,  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1,
  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1,  -1,
  -1, 1,  1,  1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1,  -1,  -1,
  -1, -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  -1,  -1,  -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  -1,  -1,  -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  -1,  -1,  -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  -1,  -1,  1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  -1,  1,  1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  -1,  1,  1,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, -1,  1,  1,  1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,
    -1, 1,  1,  1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
]


    y = teste_perceptron(x, w, b)

    # Exibir o resultado do perceptron
    estimativa = None

    for i, digito in enumerate(digitos.keys()):
        print(f"Saída para a dígito {digito}: y = {y[i]}")
        if y[i] == 1:
            estimativa = digitos[digito]

    print("Entrada:")
    print_digito(x)

    if estimativa is None:
        print("Não foi possível identificar a dígito")
    else:
        print("Saída estimada:")
        print_digito(estimativa)


# Função para imprimir a digito
def print_digito(digito):
    for i in range(20):
        for j in range(20):
            print('#' if digito[i*20 + j] == 1 else ' ', end='')
        print()


# Função de ativação
def ativacao(yliq): 
    if yliq >= 0:  # adotei θ = 0
        return 1
    else:
        return -1
    

# Função para treinar o perceptron
def algoritmo_perceptron(digitos, taxa_aprendizado):
    num_digitos = len(digitos)
    w = [[0] * 400 for i in range(num_digitos)]
    b = [0] * num_digitos
    alfa = taxa_aprendizado

    while True:
        pesos_mudaram = False
        
        for i, (digito, valores) in enumerate(digitos.items()):
            x = valores
            
            t = []
            for j in range(num_digitos):
                if j == i:
                    t.append(1)
                else:
                    t.append(-1)

            for j in range(num_digitos):
                yliq = sum(w[j][k] * x[k] for k in range(400)) + b[j]
                y = ativacao(yliq)

                if y != t[j]:
                    pesos_mudaram = True
                    for k in range(400):
                        w[j][k] += alfa * x[k] * t[j]
                    b[j] += alfa * t[j]
        
        if not pesos_mudaram:
            break

    return w, b


# Função para testar o perceptron
def teste_perceptron(x, w, b):
    y = []
    for i in range(len(w)):
        yliq = sum(w[i][k] * x[k] for k in range(400)) + b[i]
        y.append(ativacao(yliq))
    return y


if __name__ == '__main__':
    main()
