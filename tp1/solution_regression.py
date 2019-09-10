# -*- coding: utf-8 -*-

#####
# VosNoms (Matricule) .~= À MODIFIER =~.
###

import numpy as np
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        phi_x = x

        if type(x) == int:
            phi_x = x**np.arange(1, self.M+1)

        elif type(x) == list:
            phi_x = [[]]
            for elem_x in x:
                phi_x = np.append(phi_x, [elem_x**np.arange(1, self.M+1)])
            N = len(x)
            phi_x.shape = (N,self.M)



        else:
            # WARNING
            print("warning temporaire")


        
        return phi_x

    def recherche_hyperparametre(self, X, t):
        """
        Validation croisee de type "k-fold" pour k=10 utilisee pour trouver la meilleure valeur pour
        l'hyper-parametre self.M.

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        # AJOUTER CODE ICI

        self.M = 1
        erreur_min = float("inf")
        k = 10
        
        for M_actuel in range(0,10):
            for lamb_actuel in range(0,10):
                for j in range(0,k):

                     X_train, X_valid, t_train, t_valid = train_test_split(X, t, test_size=0.20)
                     entrainement(X_train,t_train)

                     pred = prediction(X_valid)
                     erreur_moy = erreur(t_valid, pred)

                     if erreur_moy < erreur_min:
                         erreur_min = erreur_moy
                         self.M = M_actuel
                         self.lamb = lamb_actuel




        

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        #AJOUTER CODE ICI
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)
        self.w = [0, 1]

        if using_sklearn == False:





        elif using_sklearn == True:





        else:
            print("Mauvaise valeur de 'using_sklearn', doit etre un booleen.")

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI
        return 0.5

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI
        return 0.0
