l_erreur_moyenne_folds = []
l_parametre = []
if self.noyau == "rbf":
    for valeur_sigma_carre in np.linspace(0.000000001, 2, 10):
        for valeur_lamb in np.linspace(0.000000001, 2, 10):
            l_erreur_moyenne_folds.append(self.k_folds(x_tab, t_tab))
            l_parametre.append((valeur_sigma_carre,valeur_lamb))
    id_minimun_erreur_moy = np.argmin(l_erreur_moyenne_folds)
    self.sigma_square, self.lamb = l_parametre[id_minimun_erreur_moy]

if self.noyau == "lineaire":
    for valeur_lamb in np.linspace(0.000000001, 2, 10):
        l_erreur_moyenne_folds.append(self.k_folds(x_tab, t_tab))
        l_parametre.append(valeur_lamb)
    id_minimun_erreur_moy = np.argmin(l_erreur_moyenne_folds)
    self.lamb = l_parametre[id_minimun_erreur_moy]

if self.noyau == "polynomial":
    for valeur_c in np.linspace(0, 5, 6):
        for valeur_lamb in np.linspace(0.000000001, 2, 10):
            for valeur_M in np.linspace(2,6, 4):
                l_erreur_moyenne_folds.append(self.k_folds(x_tab, t_tab))
                l_parametre.append((valeur_lamb, valeur_c, valeur_M))
    id_minimun_erreur_moy = np.argmin(l_erreur_moyenne_folds)
    self.lamb, self.c, self.M = l_parametre[id_minimun_erreur_moy]

if self.noyau == "sigmoidal":
    for valeur_b in np.linspace(0.00001, 0.01, 10):
        for valeur_lamb in np.linspace(0.000000001, 2, 10):
            for valeur_d in np.linspace(0.00001, 0.01, 10):
                l_erreur_moyenne_folds.append(self.k_folds(x_tab, t_tab))
                l_parametre.append((valeur_lamb, valeur_b, valeur_d))
    id_minimun_erreur_moy = np.argmin(l_erreur_moyenne_folds)
    self.lamb, self.b, self.d = l_parametre[id_minimun_erreur_moy]

	
	
	
