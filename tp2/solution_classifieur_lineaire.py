# -*- coding: utf-8 -*-

#####
# Vos Noms (VosMatricules) .~= À MODIFIER =~.
####

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.]) # paramètre aléatoire
        self.w_0 = -5.              # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')
            # AJOUTER CODE ICI
            
            #p = 1/N*sum(tn) = N1/N = N1/(N1+N2)
            #p=p(C1), p(C2)=1-p
            N = len(t_train)
            N1 = sum(t_train)
            N2 = N-N1
            p = N1/N
            
            #mu_1 = 1/N1*sum(tn*xn)
            mu_1 = 1/N1*sum(np.dot(t_train,x_train))
            
            #mu_2 = 1/N2*sum((1-tn)*xn)
            mu_2 = 1/N2*sum(np.dot((1-t_train),x_train))

            #S1 = 1/N1*sum(xn-mu_1)*t(xn-mu_1)
            S1 = 1/N1*sum(np.dot((x_train-mu_1),np.transpose(x_train-mu_1)))
            
            #S2 = 1/N2*sum(xn-mu_2))*t(xn-mu_2))
            S2 = 1/N2*sum(np.dot((x_train-mu_2),np.transpose(x_train-mu_2)))

            #sigma = N1/N*S1 + N2/N*S2
            sigma = N1/N*S1 + N2/N*S2
            
            #self.w = sigma^(-1)*(mu_1-mu_2)
            self.w = np.dot(np.linalg.inv(sigma),(mu_1-mu_2))
            
            #self.w_0 = -0.5*t(mu_1)*sigma^(-1)*mu_1 + 0.5*t(mu_2)*sigma^(-1)*mu_2 + ln(p(C1)/p(C2))
            self.w_0 = -0.5*np.dot(np.dot(np.transpose(mu_1),np.linalg.inv(sigma)),mu_1) + 0.5*np.dot(np.dot(np.transpose(mu_2),np.linalg.inv(sigma)),mu_2)
            
        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')
            x_train = np.append(x_train, [[-1]]*len(x_train), axis=1)
            t_train = np.where(t_train==0, -1, t_train)
            self.w = np.zeros(len(x_train[0]))

            eta0 = 0.001
            n_iter = 1000

            for i in np.nditer(n_iter):
                for i, x in enumerate(x_train):
                    if(np.dot(x_train[i], self.w) * t_train[i]) <= 0:
                        self.w = self.w + eta0 * x_train[i] * t_train[i]
            
            self.w_0 = - self.w[-1] + 0.15

            x_train = np.delete(x_train, 2, 1)
            t_train = np.where(t_train==-1, 0, t_train)
            self.w = np.delete(self.w, -1, 0)

        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 + penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')
            clf = Perceptron(penalty='l2',eta0=0.001)
            clf.fit(x_train,t_train)
            self.w = clf.coef_[0]
            self.w_0 = clf.intercept_

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """
        frontiere = np.dot(self.w, x)

        if(frontiere > 0):
            y = 1
        elif(frontiere < 0):
            y = 0
        else:
            print("Le point x est sur la frontiere")
        return y

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """
        if t != prediction:
            return 1
        else:
            return 0

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
