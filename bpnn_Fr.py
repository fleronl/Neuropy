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

# Calcule un chiffre al�atoire compris entre:  a <= rand < b
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
    return math.tanh(x)            # 500 iterations conseill�
    #return math.atan(x)             # 150 iterations conseill�
    #return 1/(1 + math.exp(-x))

# La derive de la fonction sigmoid, pour notre cas la derive de tan(x)
def dsigmoid(y):
    return 1.0 - y**2
    #return 1 / (1.0 - y**2)

class NN:
    def __init__(self, ni, nh, no):

        print ("--- INITIALISATION DU RESEAU ---")

        #Nombre de neurones en entr�e, cach� et en sortie
        self.ni = ni + 1 # +1 Pour le neurone de biais qui est positionn� � la fin
        self.nh = nh
        self.no = no
        print ("Nbre de neurones en entr�e [{0}] dont [1] biais, cach�s [{1}], sortie [{2}] \n".format(self.ni, self.nh, self.no))

        # activation des neurones
        # Cr�ation de 3 listes contenant le nombre de neurones dans chaque couche
        self.ai = [1.0]*self.ni     # ni pour entr�
        self.ah = [1.0]*self.nh     # nh pour cach�
        self.ao = [1.0]*self.no     # no pour sorti
        print ("Activation par d�faut des neurones en entr�e [{0}], cach�s [{1}], sortie [{2}] \n".format(self.ai, self.ah, self.ao))


        # Cr�ation des poids wi entre les neurones en entr�e et cach�s et entre les neurones cach�s et en sortie
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        print ("Cr�ation des poids wi entre entr�e/cach� [{0}], cach�/sortie [{1}] \n".format(self.wi, self.wo))

        # Enregistre les valeurs al�atoires dans la liste W (Poids du neurone)
        # Poids W entre les neurones input et Hide puis entre Hide et Output
        print ("Poids al�atoire wi entre les neurones en entr�e et cach�")
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
                print ("self.w[{0}][{1}]: {2}".format(i, j, self.wi[i][j]))
        print ("\n")

        print ("Poids al�atoire wi entre les neurones cach�s et de sortie")
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
                print ("self.w[{0}][{1}]: {2}".format(j, k, self.wi[j][k]))
        print ("\n")

        # M�morisation des poids au temps n-1
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        #print "M�morisation des anciens poids entre entr�e/cach� [{0}], cach�/sorti� [{1}] \n".format(self.ci, self.co)

    # Mise � jour des neurones
    def update(self, inputs):

        # Si le nombre entr� ne correspond pas � la matrice
        if len(inputs) != self.ni-1:
            raise ValueError("Mauvaise valeur dans le nombre d'entr�es")

        # Calcul des neurones en entr�e
        #print "\n----- VALEUR DES 'ai'"
        for i in range(self.ni-1):
            self.ai[i] = sigmoid(inputs[i])
            #self.ai[i] = inputs[i]
            #print "-> ai[{0}]: {1}".format(i, self.ai[i])

        # Calcul des neurones cach�s
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

    # Calcul des poids par r�tro-propagation et mise � jour
    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError("Mauvaise valeur entre le nombre de neurones de sortie et la liste d'essais nomm� pat")

        #Passe en arri�re
        # Calcul de l'erreur et du delta entre la valeur cherch�e et celle calcul�e en sortie
        output_deltas = [0.0] * self.no                         #Construction d'une liste contenant no valeurs � 0.0
        for k in range(self.no):
            error = targets[k]-self.ao[k]                       #Erreur = Diff�rence entre cherch� et calcul�
            output_deltas[k] = dsigmoid(self.ao[k]) * error     #l'ecart = d�riv� de calcul� multipli� par la diff�rence

        # Calcul de l'erreur et du delta entre la valeur cherch�e et celle calcul�e pour la couche cach�e
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        #Passe en avant *************************************************
        # Mise � jour des poids de la couche de sortie
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # Mise � jour des poids de la couche d'entr�e
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
        print("R�seau apr�s apprentissage")
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

        print ("--- APPRENTISSAGE DU RESEAU par mise � jour des poids et r�tro-propagation---")

        # N: learning rate, le taux d'apprentissage ou facteur d'inertie afin d'�viter les minimums locaux.
        # M: momentum factor, le 'facteur de m�morisation ??' Le pas variable ??
	# pour essayer d'am�liorer la vitesse de convergence,
	# une variante couramment utilis�e consiste � pond�rer la modification des poids en fonction du nombre d'it�rations d�ja effectu�
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                #print "--->", self.ao[:]
                error = error + self.backPropagate(targets, N, M)

            print ("It�ration :", i + 1)
            
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
    # Cr�ation d'un r�seau de neurones avec 2 entr�es, 2 cach�s et une sortie
    n = NN(2, 2, 1)

    # Entrainement du r�seau avec la liste d'essais ci dessus nomm� pat
    n.train(pat)

    # Test du r�seau
    n.test(pat)

    # R�sume et affichage des poids W calcul�s
    print ("\nLes poids W:")
    n.weights()

    # R�sume et affichage des poids N calcul�s
    #print ("\nLes poids N:")
    #n.neuron()

if __name__ == '__main__':
    demo()
