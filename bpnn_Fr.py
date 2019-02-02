# -*- coding: latin1 -*-
# Back-Propagation Neural Networks
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

# Traduction Laurent Fleron
# laurent.fleron@ac-reims.fr
# Mise a jour 27/12/2013

import math
import random
import string

random.seed(0)

# Calcule un chiffre aléatoire compris entre:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Construit une liste m d'indice i
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# La fonction d'activation en remplacement de la sigmoid 1/(1+e^-x) par tanh(x) ou atan(x)
def sigmoid(x):
    return math.tanh(x)            # 500 iterations conseillé
    #return math.atan(x)             # 150 iterations conseillé
    #return 1/(1 + math.exp(-x))

# La derive de la fonction sigmoid, pour notre cas la derive de tan(x)
def dsigmoid(y):
    return 1.0 - y**2
    #return 1 / (1.0 - y**2)

class NN:
    def __init__(self, ni, nh, no):

        print ("--- INITIALISATION DU RESEAU ---")

        #Nombre de neurones en entrée, caché et en sortie
        self.ni = ni + 1 # +1 Pour le neurone de biais qui est positionné à la fin
        self.nh = nh
        self.no = no
        print ("Nbre de neurones en entrée [{0}] dont [1] biais, cachés [{1}], sortie [{2}] \n".format(self.ni, self.nh, self.no))

        # activation des neurones
        # Création de 3 listes contenant le nombre de neurones dans chaque couche
        self.ai = [1.0]*self.ni     # ni pour entré
        self.ah = [1.0]*self.nh     # nh pour caché
        self.ao = [1.0]*self.no     # no pour sorti
        print ("Activation par défaut des neurones en entrée [{0}], cachés [{1}], sortie [{2}] \n".format(self.ai, self.ah, self.ao))


        # Création des poids wi entre les neurones en entrée et cachés et entre les neurones cachés et en sortie
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        print ("Création des poids wi entre entrée/caché [{0}], caché/sortie [{1}] \n".format(self.wi, self.wo))

        # Enregistre les valeurs aléatoires dans la liste W (Poids du neurone)
        # Poids W entre les neurones input et Hide puis entre Hide et Output
        print ("Poids aléatoire wi entre les neurones en entrée et caché")
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
                print ("self.w[{0}][{1}]: {2}".format(i, j, self.wi[i][j]))
        print ("\n")

        print ("Poids aléatoire wi entre les neurones cachés et de sortie")
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
                print ("self.w[{0}][{1}]: {2}".format(j, k, self.wi[j][k]))
        print ("\n")

        # Mémorisation des poids au temps n-1
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        #print "Mémorisation des anciens poids entre entrée/caché [{0}], caché/sortié [{1}] \n".format(self.ci, self.co)

    # Mise à jour des neurones
    def update(self, inputs):

        # Si le nombre entré ne correspond pas à la matrice
        if len(inputs) != self.ni-1:
            raise ValueError("Mauvaise valeur dans le nombre d'entrées")

        # Calcul des neurones en entrée
        #print "\n----- VALEUR DES 'ai'"
        for i in range(self.ni-1):
            self.ai[i] = sigmoid(inputs[i])
            #self.ai[i] = inputs[i]
            #print "-> ai[{0}]: {1}".format(i, self.ai[i])

        # Calcul des neurones cachés
        #print "\n----- VALEUR DES 'wi' puis des 'ah'"
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
                #print "w[{0}][{1}]: {2}".format(i, j, self.wi[i][j])
            self.ah[j] = sigmoid(sum)
            #print "-> ah[{0}]: {1}".format(j, self.ah[j])

        # Calcul des neurones de sortie
        #print "\n----- VALEUR DES 'wo' puis des 'ao'"
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
                #print "w[{0}][{1}]: {2}".format(j, k, self.wo[j][k])
            self.ao[k] = sigmoid(sum)
            #print "-> ao[{0}]: {1}".format(k, self.ao[k])


        return self.ao[:]

    # Calcul des poids par rétro-propagation et mise à jour
    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError("Mauvaise valeur entre le nombre de neurones de sortie et la liste d'essais nommé pat")

        #Passe en arrière
        # Calcul de l'erreur et du delta entre la valeur cherchée et celle calculée en sortie
        output_deltas = [0.0] * self.no                         #Construction d'une liste contenant no valeurs à 0.0
        for k in range(self.no):
            error = targets[k]-self.ao[k]                       #Erreur = Différence entre cherché et calculé
            output_deltas[k] = dsigmoid(self.ao[k]) * error     #l'ecart = dérivé de calculé multiplié par la différence

        # Calcul de l'erreur et du delta entre la valeur cherchée et celle calculée pour la couche cachée
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        #Passe en avant *************************************************
        # Mise à jour des poids de la couche de sortie
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # Mise à jour des poids de la couche d'entrée
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calcule de l'erreur
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        print("Réseau après apprentissage")
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for j in range(self.ni):
            for k in range(self.nh):
                print ("w[{0}][{1}]: {2}".format(j, k, self.wi[j][k]))

        print('\nOutput weights:')
        for j in range(self.nh):
            for k in range(self.no):
                print ("w[{0}][{1}]: {2}".format(j, k, self.wo[j][k]))

    #def neuron(self):
        #print('Input neurons:')
        #for i in range (len(self.ai)):
            #print "ai[{0}]: {1}".format(i, self.ai[i])
        #print (self.ai)

        #print('hiden neurons:')
        #for j in range(len(self.ah)):
            #print "ah[{0}]: {1}".format(j, self.ah[j])
        #print(self.nh)

        #print('out neurons:')
        #for j in range(len(self.ao)):
            #print "ao[{0}]: {1}".format(j, self.ao[j])
        #print(self.nh)

    # Apprentissage du reseau
    def train(self, patterns, iterations=200, N=0.5, M=0.1):

        print ("--- APPRENTISSAGE DU RESEAU par mise à jour des poids et rétro-propagation---")

        # N: learning rate, le taux d'apprentissage ou facteur d'inertie afin d'éviter les minimums locaux.
        # M: momentum factor, le 'facteur de mémorisation ??' Le pas variable ??
	# pour essayer d'améliorer la vitesse de convergence,
	# une variante couramment utilisée consiste à pondérer la modification des poids en fonction du nombre d'itérations déja effectué
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                #print "--->", self.ao[:]
                error = error + self.backPropagate(targets, N, M)

            print ("Itération :", i + 1)
            
            for p in patterns:
                print(p[0], '->', self.update(p[0]))

            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    # Apprentissage du reseau pour la fonction logique 'XOR'
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [1]]
    ]

    # Perceptron simple couche
    # Création d'un réseau de neurones avec 2 entrées, 2 cachés et une sortie
    n = NN(2, 2, 1)

    # Entrainement du réseau avec la liste d'essais ci dessus nommé pat
    n.train(pat)

    # Test du réseau
    n.test(pat)

    # Résume et affichage des poids W calculés
    print ("\nLes poids W:")
    n.weights()

    # Résume et affichage des poids N calculés
    #print ("\nLes poids N:")
    #n.neuron()

if __name__ == '__main__':
    demo()
