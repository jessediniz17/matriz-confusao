# Resumo do Projeto: Classificação de Imagens CIFAR-10

Este projeto foca na classificação de imagens utilizando o conjunto de dados CIFAR-10, uma tarefa fundamental em Visão Computacional. O objetivo é treinar uma Rede Neural Convolucional (CNN) para categorizar imagens em 10 classes distintas (por exemplo, avião, automóvel, pássaro, etc.).

## Conjunto de Dados: CIFAR-10

O CIFAR-10 é um conjunto de dados de imagens pequenas e coloridas. Ele consiste em:
- **Imagens:** 60.000 imagens 32x32 pixels coloridas (RGB).
- **Divisão:** 50.000 imagens de treinamento e 10.000 imagens de teste.
- **Classes:** 10 classes, com 6.000 imagens por classe.

As imagens foram normalizadas para valores entre 0 e 1 durante o pré-processamento.

## Arquitetura do Modelo: Rede Neural Convolucional (CNN)

O modelo utilizado é uma CNN sequencial composta por:
- **Camadas Convolucionais:** Múltiplas camadas `Conv2D` com ativação ReLU para extrair características.
- **Camadas de Pooling:** Camadas `MaxPooling2D` para redução dimensional e invariância a pequenas translações.
- **Camada de Flatten:** Converte a saída 3D das camadas convolucionais em um vetor 1D.
- **Camadas Densas (Fully Connected):** Uma camada `Dense` oculta com ativação ReLU e uma camada de saída `Dense` com 10 neurônios (um para cada classe).

O modelo foi compilado com o otimizador 'adam', função de perda 'sparse_categorical_crossentropy' e métrica de 'accuracy'.

## Treinamento e Avaliação

O modelo foi treinado por 5 épocas, utilizando `TensorBoard` para monitoramento e um `LambdaCallback` para registrar a matriz de confusão a cada época. Após o treinamento, as previsões foram geradas para o conjunto de teste e as seguintes métricas foram calculadas:

- **Acurácia (Accuracy):** 8.20% (Nota: A acurácia calculada via `(VP + VN) / N` pode ser enganosa para classificação multiclasse desbalanceada ou quando a interpretação de VN/FP não é adaptada corretamente a cada classe, e a acurácia global tende a ser `sum(diag(con_mat)) / total_samples`).
- **Sensibilidade (Recall):** 10.00% (`VP / (VP + FN)`)
- **Especificidade (Specificity):** 90.00% (`VN / (FP + VN)`)
- **Precisão (Precision):** 10.00% (`VP / (VP + FP)`)
- **F-Score:** 10.00% (`2 * (Precision * Recall) / (Precision + Recall)`)

## Análise da Matriz de Confusão

A matriz de confusão (normalizada) mostra que a maioria das previsões do modelo está concentrada na classe 5. Isso indica que o modelo está tendencioso a prever a classe 5 para todas as entradas, resultando em um desempenho muito baixo para as outras classes e uma acurácia global ineficaz. Este comportamento sugere que o modelo não aprendeu a distinguir efetivamente entre as diferentes classes do CIFAR-10, provavelmente necessitando de mais épocas de treinamento, ajuste de hiperparâmetros ou uma arquitetura de modelo mais complexa.
