# -*- coding: utf-8 -*-

#####
# Lauren Picard 19 159 731
# Julien Brosseau 19 124 617
# Antoine Gelin 19 146 158
###

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

    def noyau_rbf(self,x1,x2):
        return np.exp(-np.linalg.norm(x1-x2)**2/(2*self.sigma_square))

    def noyau_lineaire(self,x1,x2):
        return np.dot(np.transpose(x1),x2)
    
    def noyau_poly(self,x1,x2):
        return (np.dot(np.transpose(x1),x2)+self.c)**self.M
    
    def noyau_sigmoidal(self,x1,x2):
        return np.tanh(self.b*np.dot(np.transpose(x1),x2)+self.d)

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """

        K = np.zeros((x_train[:,0].size, x_train[:,0].size))
        #K = np.zeros((t_train.size,t_train.size))

        if self.noyau == 'rbf':
            for i in range(x_train[:,0].size):
                for j in range(x_train[:,0].size):
                    K[i][j] = self.noyau_rbf(x_train[i],x_train[j])

        elif self.noyau == 'lineaire':
            for i in range(x_train[:,0].size):
                for j in range(x_train[:,0].size):
                    K[i][j] = self.noyau_lineaire(x_train[i],x_train[j])
        
        elif self.noyau == 'polynomial':
            for i in range(x_train[:,0].size):
                for j in range(x_train[:,0].size):
                    K[i][j] = self.noyau_poly(x_train[i],x_train[j])
                    
        elif self.noyau == 'sigmoidal':
            for i in range(x_train[:,0].size):
                for j in range(x_train[:,0].size):
                    K[i][j] = self.noyau_sigmoidal(x_train[i],x_train[j])

        
        #K = phi*np.transpose(phi) # matrice de Gram
        #print()
        self.a = np.dot(np.linalg.inv(K+self.lamb*np.identity(K[:,0].size)),t_train)
        self.x_train = x_train
        
    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        sum=0
        if self.noyau == "rbf":
            for i in range(self.x_train[:,0].size):
                sum += self.noyau_rbf(x,self.x_train[i])*self.a[i] #p18 cours methodes a noyau
        
        elif self.noyau == "lineaire":
            for i in range(self.x_train[:,0].size):
                sum += self.noyau_lineaire(x,self.x_train[i])*self.a[i]
                
        elif self.noyau == "polynomial":
            for i in range(self.x_train[:,0].size):
                sum += self.noyau_poly(x,self.x_train[i])*self.a[i]
            
        elif self.noyau == "sigmoidal":
            for i in range(self.x_train[:,0].size):
                sum += self.noyau_sigmoidal(x,self.x_train[i])*self.a[i]
        #print("sum ", sum)
        if sum > 0.5:
            return 1
        else:
            return 0

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (t-prediction)**2

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=1 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        print("Recherche d'hyperparametres")

        erreur_moy = []
        new_hyperparametres = []

        M_min = 2
        M_max = 6
        M_step = 1
        
        l_min = 0.000000001
        l_max = 2
        l_step = 0.2

        b_min = 0.00001
        b_max = 0.01
        b_step = 0.001
        
        c_min = 0
        c_max = 5
        c_step = 0.1

        d_min = 0.00001
        d_max = 0.01
        d_step = 0.001
        
        sigma_min = 0.000000001
        sigma_max = 2
        sigma_step = 0.05

        def erreur_moyenne(self, x_tab, t_tab):
            erreur_moy = 0

            X_train, X_valid, t_train, t_valid = train_test_split(x_tab, t_tab, test_size=0.20)
            self.entrainement(X_train,t_train)
            
            for i in range(len(X_valid)):
                pred = self.prediction(X_valid[i])
                erreur_moy += self.erreur(t_valid[i], pred)
            
            return erreur_moy/len(X_valid)
        
        if self.noyau == "rbf":
            for lamb_actuel in np.arange(l_min,l_max,l_step):
                for sigma_actuel in np.arange(sigma_min,sigma_max,sigma_step):
                    erreur_moy.append(erreur_moyenne(self, x_tab, t_tab))
                    new_hyperparametres.append((lamb_actuel, sigma_actuel))
            self.lamb, self.sigma_square = new_hyperparametres[np.argmin(erreur_moy)]
            print("Lambda :", self.lamb, "| Sigma :", self.sigma_square)
        
        if self.noyau == "lineaire":
            for lamb_actuel in np.arange(l_min,l_max,l_step):
                erreur_moy.append(erreur_moyenne(self, x_tab, t_tab))
                new_hyperparametres.append((lamb_actuel))
            self.lamb = new_hyperparametres[np.argmin(erreur_moy)]
            print("Lambda :", self.lamb)
        
        if self.noyau == "polynomial":
            for lamb_actuel in np.arange(l_min,l_max,l_step):
                for c_actuel in np.arange(c_min,c_max,c_step):
                    for M_actuel in np.arange(M_min,M_max,M_step):
                        erreur_moy.append(erreur_moyenne(self, x_tab, t_tab))
                        new_hyperparametres.append((lamb_actuel, c_actuel, M_actuel))
            self.lamb, self.c, self.M = new_hyperparametres[np.argmin(erreur_moy)]
            print("Lambda :", self.lamb, "| C :", self.c, "| M :", self.M)
        
        if self.noyau == "sigmoidal":
            for lamb_actuel in np.arange(l_min,l_max,l_step):
                for b_actuel in np.arange(b_min, b_max, b_step):
                    for d_actuel in np.arange(d_min, d_max, d_step):
                        erreur_moy.append(erreur_moyenne(self, x_tab, t_tab))
                        new_hyperparametres.append((lamb_actuel, b_actuel, d_actuel))
            self.lamb, self.b, self.d = new_hyperparametres[np.argmin(erreur_moy)]
            print("Lambda :", self.lamb, "| B :", self.b, "| D:", self.d)

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()
