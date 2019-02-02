# -*- coding: cp1252 -*-

class NeuroGen(object):

    """Classe Parente définissant un neurone caractérisé par :
    - sa localisation (E)ntré, (S)ortie, (I)nterne) ;
    - son poids Xi """

    NbrNeuro = 0     #Le compteur d'objets NeuroGen = 0
    TetajDef = 0.22  #Valeur par défaut de la fonction Tetaj
    
    # _init_ est une méthode de l'instance objet qui définit notre classe Neurone
    # self est un attribut de la méthode _init_
    def __init__(self, Localisation, PoidsXi=0):
        """ Constructeur de la classe NeuroGen"""
        self.Local = Localisation
        self.Xi = PoidsXi

        """ On compte le nombre de neurones crées """
        NeuroGen.NbrNeuro += 1
        
    def __repr__(self):
        """ Constructeur de l'objet permettant de le representer formaté """
        return " Neurone localisé en couche ({}) avec un poids Xi ({})".format(self.Local, self.Xi)
    

class NeuroInt(NeuroGen):
    """ Classe Fille définissant un neurone interne caractérisé par :
    - son poids Wij ;
    - sa fonction de combinaison (la somme pondérée) Netj ;
    - son seuil Tetaj ;
    - sa fonction d'activatio Phi
    Héritage : ;
    - la Localisation ;
    - le PoidsXi"""
    
    def __init__(self, Localisation, PoidsXi=0, PoidsWij=10, ValNetj=0, ValTetaj=0, ValPhi=0):
        """..."""
        NeuroGen.__init__(self, Localisation, PoidsXi)
        self.Wij = PoidsWij
        self.Netj = ValNetj
        self.Tetaj = ValTetaj
        self.Phi = ValPhi

    def __repr__(self):
        """ Constructeur de l'objet permettant de le representer formaté """
        return " Neurone Interne avec un poids Xi ({}) ET un poids Wij ({})".format(self.Xi, self.Wij)
    
    
