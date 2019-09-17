# -*- coding: utf-8 -*-

#####
# VosNoms (Matricule) .~= À MODIFIER =~.
###

import numpy as np
import random
from sklearn.linear_model import Ridge
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

        # if type(x) == int:
        #     phi_x = x**np.arange(1, self.M+1)

        if type(x) == np.ndarray:
            phi_x = [[]]
            for elem_x in x:
                phi_x = np.append(phi_x, [elem_x**np.arange(1, self.M+1)])
            N = len(x)
            phi_x.shape = (N,self.M)



        else: 

            phi_x = x**np.arange(1, self.M+1)
            


        
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
        print("recherche d'HP")

        self.M = 1
        erreur_min = float("inf")
        k = 10

        M_min = 0
        M_max = 10
        M_step = 1
        
        l_min = 0
        l_max = 10
        l_step = 1
        
        for M_actuel in range(M_min,M_max,M_step):
            for lamb_actuel in range(l_min,l_max,l_step):
                for j in range(0,k):

                    X_train, X_valid, t_train, t_valid = train_test_split(X, t, test_size=0.20)
                    self.entrainement(X_train,t_train,True)

                    erreur_moy = 0
                    for x in X_valid:
                        pred = self.prediction(x)
                        erreur_moy = self.erreur(t_valid, pred)
                    erreur_moy = erreur_moy/len(X_valid)

                    print("M: ",M_actuel,", Lambda = ",lamb_actuel,", j: ",j,", erreur: ",erreur_moy)

                    if erreur_moy < erreur_min:
                        erreur_min = erreur_moy
                        self.M = M_actuel
                        self.lamb = lamb_actuel


        print("M: ",self.M)
        print("lanbda: ",self.lamb)


        

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

        w = (λI + Φ^T Φ)^− 1 * Φ^T t.

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        #AJOUTER CODE ICI
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)
        #self.w = [0, 1]

        if not using_sklearn:

            phi_x = self.fonction_base_polynomiale(X)
            phi_x_t = np.transpose(phi_x)
            dim_I = phi_x.shape[1]
            
            
            # self.w = np.dot(((np.dot(phi_x_t,phi_x)+self.lamb)**-1),np.dot(phi_x_t,t))
            #self.w = np.dot((np.linalg.inv(np.dot(phi_x_t,phi_x) + self.lamb*np.identity(dim_I)),np.dot(phi_x_t,t)))
            
            membre1 = np.linalg.inv(self.lamb*np.identity(dim_I) + np.dot(phi_x_t,phi_x))
            membre2 = np.dot(phi_x_t,t)
            self.w = membre1*membre2

            print(self.w)
            #np.linalg.solve()




        elif using_sklearn :
            #X = X.reshape(-1,1)
            reg = Ridge(alpha=self.lamb)
            reg.fit(phi_x,t)
            self.w = []
            self.w = np.append(self.w, reg.intercept_)
            self.w = np.append(self.w, reg.coef_)
            print(self.w)




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
        #phi_x = [1]
        phi_x=[]
        phi_x = np.append(phi_x, self.fonction_base_polynomiale(x))
        y = np.dot(np.transpose(self.w),phi_x)
        return y


    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI
        return (t-prediction)**2
