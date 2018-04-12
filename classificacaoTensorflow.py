# Utilizando tensorflow e regressão logística para classificar algorismos numéricos
# Base de dados: mnist database, disponível em http://yann.lecun.com/exdb/mnist/
# Um conjunto de dados com dígitos numéricos escritos a mão
# Esse conjunto de dados já está rotulado


# Com Regressão logística, buscamos uma função que nos diga qual é a probabilidade de um elemento pertencer a uma classe. 
# A aprendizagem supervisionada é configurada como um processo iterativo de otimização dos pesos. Estes são então modificados com base no desempenho do modelo.
# O objetivo é minimizar a função de perda, que indica o grau em que o comportamento do modelo se desvia do desejado. 
# O desempenho do modelo é verificado em um conjunto de teste, consistindo em imagens diferentes das de treinamento.

# Os passos básicos do treinamento são os seguintes: 

# 1- Os pesos são inicializados com valores aleatórios no início do treinamento. 
# 2- Para cada elemento do conjunto de treino é calculado o erro, ou seja, a diferença entre a saída desejada e a saída real. Este erro é usado para ajustar os pesos. 
# 3- O processo é repetido, em uma ordem aleatória, em todos os exemplos do conjunto de treinamento até que o erro em todo o conjunto 
# de treinamento não seja inferior a um certo limite, ou até que o número máximo de iterações seja atingido.



# Importando pacotes
import tensorflow as tf
import matplotlib.pyplot as plt

import input_data # módulo desenvolvido pela equipe de desenvolvimento do tensorflow para realizar o download banco de dados tensorflow
# input_data.py está disponível no link abaixo
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py

# Importando dataset MINST
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Parâmetros
learning_rate = 0.01 #taxa de aprendizagem - Como o modelo vai seguir o caminho de aprendizagem
training_epochs = 25 #número de iterações realizadas pelo modelo
batch_size = 100 # quantidade de dados utilizados em cada passada
display_step = 1 # intervalo de passos que serão apresentados na tela

# Placeholders
# O problema consiste em atribuir um valor de probabilidade para cada uma das possíveis classes de membros (os números de 0 a 9). 
x = tf.placeholder("float", [None, 784]) # Variável preditora - Dados de imagens do mnist com shape 28*28 = 784
y = tf.placeholder("float", [None, 10])  # Variável target - Classe - 0-9 digits (10 classes)


# Criando o Modelo

# Construindo o Modelo
# Para atribuir probabilidades a cada imagem, usaremos a chamada função de ativação softmax.
# A função softmax é especificada em duas etapas principais: 
# 1-Calcular a evidência de que uma determinada imagem pertence a uma determinada classe.
# 2-Converter a evidência em probabilidades de pertencer a cada uma das 10 classes possíveis.

# Para uma determinada imagem, podemos avaliar a evidência para cada classe i simplesmente multiplicando o tensor W pelo tensor de entrada x. 
# Em geral, os modelos incluem um parâmetro extra que representa o viés (bias - b), o que indica um certo grau de incerteza. 
# Significa que para cada i (de 0 a 9) temos elementos de matriz Wi (784), onde cada elemento j da matriz é multiplicado 
# pelo componente correspondente j da imagem de entrada (784 partes).

# Definindo os pesos, variáveis W e b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Função de Ativação Softmax
# A função tf.nn.softmax do TensorFlow fornece uma saída baseada na probabilidade a partir do tensor de evidência de entrada. 
# Uma vez implementado o modelo, podemos especificar o código necessário para encontrar os pesos W  e b, através do algoritmo de treinamento iterativo. 
# Em cada iteração, o algoritmo recebe os dados de treinamento, aplica a função e compara o resultado com o esperado (observado).
# Na teoria da probabilidade, a saída da função softmax pode ser usada para representar uma distribuição categórica.
# A função de ativação softmax é usada em redes neurais de classificação. Ela força a saída de uma rede neural a representar a probabilidade dos dados serem 
# de uma das classes definidas. Sem ela as saídas dos neurônios são simplesmente valores numéricos onde o maior indica a classe vencedora.
activation = tf.nn.softmax(tf.matmul(x, W) + b) 


# Minimizando o erro usando cross entropy
# A fim de treinar nosso modelo, devemos definir como identificar a precisão. 
# Nosso objetivo é tentar obter valores de parâmetros W e b que minimizem o valor da métrica que indica quão ruim é o modelo.
# Diferentes métricas calculam o grau de erro entre a saída desejada e as saídas de dados de treinamento. 
# Uma medida comum de erro é o erro quadrático médio ou a Distância Euclidiana Quadrada. No entanto, existem algumas descobertas de pesquisa que sugerem usar outras 
# métricas para uma rede neural. Neste exemplo, usamos a chamada função de erro de entropia cruzada. Ele é definido como:
cross_entropy = y*tf.log(activation)

# Para minimizar cross_entropy, podemos usar a seguinte combinação de tf.reduce_mean e tf.reduce_sum para construir a função de custo:
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices = 1))

# Otimizando a Cost Function
# Em seguida, devemos minimizá-lo usando o algoritmo de otimização de descida de gradiente:
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

# Configurações do Plot
avg_set = []
epoch_set=[]

# Inicializando as variáveis
init = tf.global_variables_initializer()

# Sessão
with tf.Session() as sess:
    sess.run(init)

    # Ciclo de treinamento
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        # Loop por todas as iterações (batches)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            # Fit training usando batch data
            sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
            
            # Computando average loss
            avg_cost += sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys})/total_batch
        
        # Display logs por epoch
        if epoch % display_step == 0:
            print ("Epoch: ", '%04d' % (epoch+1), "custo = ", "{:.9f}".format(avg_cost))
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    print ("Treinamento concluído!")

    # Como podemos observar, durante a fase de treinamento a função de custo é minimizada. No final do teste, mostramos quão preciso é o modelo implementado:
    plt.plot(epoch_set,avg_set, 'o', label = 'Regressão Logística - Fase de Treinamento')
    plt.ylabel('Custo')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Testando o Modelo
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    
    # Calculando a Acurácia
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Acurácia do Modelo: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))



