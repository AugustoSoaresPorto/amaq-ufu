from tabulate import tabulate

def main():
    # Entrada de dados
    print("Lista de funções lógicas suportadas:")
    print("* F0: Constante 0 (0000)")
    print("* F1: A AND B (0001)")
    print("* F2: A AND NOT B (0010)")
    print("* F3: A (0011)")
    print("* F4: NOT A AND B (0100)")
    print("* F5: B (0101)")
    print("* F6: A XOR B (0110)")
    print("* F7: A OR B (0111)")
    print("* F8: A NOR B (1000)")
    print("* F9: A XNOR B (1001)")
    print("* F10: NOT B (1010)")
    print("* F11: B -> A (1011)")
    print("* F12: NOT A (1100)")
    print("* F13: A -> B (1101)")
    print("* F14: A NAND B (1110)")
    print("* F15: Constante 1 (1111)")
    print("* F16: Executa todas as funções acima")
    
    # Seleção de função
    opt = (input("SELECIONE UMA FUNÇÃO SEGUINDO O MODELO \"FX\":")).upper()
    match opt:
        case "F0":
            hebb([-1,-1,-1,-1], "Constante 0")
        case "F1":
            hebb([1,-1,-1,-1], "A AND B")
        case "F2":
            hebb([-1,1,-1,-1], "A AND NOT B")
        case "F3":
            hebb([1,1,-1,-1], "A")
        case "F4":
            hebb([-1,-1,1,-1], "NOT A AND B")
        case "F5":
            hebb([1,-1,1,-1], "B")
        case "F6":
            hebb([-1,1,1,-1], "A XOR B")
        case "F7":
            hebb([1,1,1,-1], "A OR B")
        case "F8":
            hebb([-1,-1,-1,1], "A NOR B")
        case "F9":
            hebb([1,-1,-1,1], " A XNOR B")
        case "F10":
            hebb([-1,1,-1,1], "NOT B")
        case "F11":
            hebb([1,1,-1,1], "B -> A")
        case "F12":
            hebb([-1,-1,1,1], "NOT A")
        case "F13":
            hebb([1,-1,1,1], "A -> B")
        case "F14":
            hebb([-1,1,1,1], "A NAND B")
        case "F15":
            hebb([1,1,1,1], "Constante 1")
        case "F16":
            hebb([-1,-1,-1,-1], "Constante 0")
            hebb([1,-1,-1,-1], "A AND B")
            hebb([-1,1,-1,-1], "A AND NOT B")
            hebb([1,1,-1,-1], "A")
            hebb([-1,-1,1,-1], "NOT A AND B")
            hebb([1,-1,1,-1], "B")
            hebb([-1,1,1,-1], "A XOR B")
            hebb([1,1,1,-1], "A OR B")
            hebb([-1,-1,-1,1], "A NOR B")
            hebb([1,-1,-1,1], " A XNOR B")
            hebb([-1,1,-1,1], "NOT B")
            hebb([1,1,-1,1], "B -> A")
            hebb([-1,-1,1,1], "NOT A")
            hebb([1,-1,1,1], "A -> B")
            hebb([-1,1,1,1], "A NAND B")
            hebb([1,1,1,1], "Constante 1")
        case _:
            print("FUNÇÃO NÃO ENCONTRADA, TENTE NOVAMENTE!")
            input("Pressione qualquer tecla para continuar...")
            main()
    
    
def hebb(target, opt):
    print("#--"*40)
        
    # Definindo variaveis e deltas de pesos
    x1 = [1, 1, -1, -1]
    x2 = [1, -1, 1, -1]
    t = [i for i in target]
    dw1 = [x1[i]*t[i] for i in range(4)]
    dw2 = [x2[i]*t[i] for i in range(4)]
    
    # Calculo de pesos
    w1 = [dw1[0], dw1[0]+dw1[1], dw1[0]+dw1[1]+dw1[2], dw1[0]+dw1[1]+dw1[2]+dw1[3]]
    w2 = [dw2[0], dw2[0]+dw2[1], dw2[0]+dw2[1]+dw2[2], dw2[0]+dw2[1]+dw2[2]+dw2[3]]
    b = [t[0], t[0]+t[1], t[0]+t[1]+t[2], t[0]+t[1]+t[2]+t[3]]
    
    # Criando tabela
    print(f'\n  TABELA DA FUNÇÃO {opt}: ')
    cabecalho = ["x1", "x2", "t", "dW1", "dW2", "db", "W1", "W2", "b"]
    tabela = [
        [x1[0], x2[0], t[0], dw1[0], dw2[0], t[0], w1[0], w2[0], b[0]],
        [x1[1], x2[1], t[1], dw1[1], dw2[1], t[1], w1[1], w2[1], b[1]],
        [x1[2], x2[2], t[2], dw1[2], dw2[2], t[2], w1[2], w2[2], b[2]],
        [x1[3], x2[3], t[3], dw1[3], dw2[3], t[3], w1[3], w2[3], b[3]],
    ]
    print(tabulate(tabela, cabecalho, tablefmt="latex"))
    
    # Pesos e bias finais
    w1f = w1[3]
    w2f = w2[3]
    bf = b[3]
    
    # Realizando teste de Hebb
    teste_hebb(x1, x2, w1f, w2f, bf, t)


def teste_hebb(x1,x2,w1,w2,b,t):
    print("\n  TESTE DE HEBB")
    
    # Encontrando yin e fyin
    yin = []
    fyin = []
    for i in range(4):
        yin.append(x1[i]*w1 + x2[i]*w2 + b)
        if yin[i] >= 0:
            fyin.append(1)
        else:
            fyin.append(-1)   
    print(f'Ao aplicar o teste de Hebb, obtemos os seguintes resultados:\n yin =  {yin} e, portanto, f(yin) = {fyin}')
    
    # Resultados
    print("\nTabela de resultados:")
    cabecalho = ["x1", "x2", "t", "yin", "fyin"]
    tabela = [
        [x1[0], x2[0], t[0], yin[0], fyin[0]],
        [x1[1], x2[1], t[1], yin[1], fyin[1]],
        [x1[2], x2[2], t[2], yin[2], fyin[2]],
        [x1[3], x2[3], t[3], yin[3], fyin[3]],
        ]
    print(tabulate(tabela, cabecalho, tablefmt="double_grid"))
    
    if fyin == t:
        print("Os resultados obtidos são iguais ao esperado, portanto, a regra de Hebb é válida.\n")
    else:
        print("Os resultados obtidos são diferentes do esperado, portanto, a regra de Hebb é inválida.\n")
    
    print("#--"*40)

if __name__ == "__main__":
    main()