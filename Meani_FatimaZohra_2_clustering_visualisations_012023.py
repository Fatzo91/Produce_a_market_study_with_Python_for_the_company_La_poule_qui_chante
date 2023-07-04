#!/usr/bin/env python
# coding: utf-8

# # Produisez une étude de marché avec Python pour l'entreprise "La poule qui chante"

# La poule qui chante, une entreprise française d’agroalimentaire. Elle souhaite se développer à l'international.

# ![Capture%20d%E2%80%99e%CC%81cran%202023-01-05%20a%CC%80%2010.24.49.png](attachment:Capture%20d%E2%80%99e%CC%81cran%202023-01-05%20a%CC%80%2010.24.49.png)

# # Sommaire :
# 
# **Partie 1 : Préparation et nettoyage des données**
# 
#  - <a href="#C1"> 1- Objectif</a>
#  - <a href="#C2"> 2- Demandes</a>
#  - <a href="#C3"> 3- Le choix des indicateurs</a>
#  - <a href="#C4"> 4- Importations des librairies</a>
#  - <a href="#C5"> 5- Importations des données</a>
#  - <a href="#C6"> 6- Préparation des données</a>
#  - <a href="#C7"> 7- Création de nouvelles variables pour notre analyse</a>
#  - <a href="#C8"> 7-1- Taux de croissance démographique sur la période 2016-2017</a>
#  - <a href="#C9"> 7-2- Taux de dépendance aux importations (TDI)</a>
#  - <a href="#C10">7-3- Taux de autosuffisance</a>
#  - <a href="#C11">7-4-  Le taux de croissance du produit intérieur brut (PIB%) (2016-2017)</a>
#  - <a href="#C12">8- Jointure</a>
#  
# **Partie 2 : Analyse exploratoire des données**
# 
#  - <a href="#C13"> 1- Corrélation entre les différentes variables</a>
#  - <a href="#C14"> 2- Matrice de corrélation</a>
#  - <a href="#C15"> 3- Vérification de la distribution des variables de tous les individus</a>
#  
# **Partie 3 : Les clusterings et les différentes visualisations associées**
# 
#  - <a href="#C16"> 1- Normalisation des données</a>
#  - <a href="#C17"> 2- Méthode de classification ascendante hiérarchique "CAH", avec un dendrogramme comme visualisation</a>
#  - <a href="#C18"> 3- La méthode des K-means</a>
#  - <a href="#C19"> 3-1- Méthode du coude</a>
#  - <a href="#C20"> 3-2- Affichage des clusters et centroïdes</a>
#  - <a href="#C21"> 3-3- Coefficient de silhoutte</a>
#  - <a href="#C22"> 4- Analyse des groupes</a>
#  - <a href="#C23"> 4-1- Découpage en classes - Matérialisation des groupes</a>
#  - <a href="#C24"> 4-2- Représentation de la distribution des variables par groupe en utilisant une boite à moustache</a>
#  - <a href="#C25"> 4-3- Croisement entre les différents clusters de pays avec les différentes variables</a>
#  - <a href="#C26"> 4-4- Corrélations entre les variables dans chaque groupe</a>
#  - <a href="#C27"> 5- Analyse des composantes principales "ACP"</a>
#  - <a href="#C28"> 6- Cercles des corrélations</a>
#  - <a href="#C29"> 7- Projections des individus</a>
#  - <a href="#C30"> 8- Exploration du cluster selectionné</a>
#  - <a href="#C31"> Conclusion</a>

# # Partie 3 : Les clusterings et les différentes visualisations associées

# # <a name="C16"> 1- Normalisation des données</a>

# In[55]:


# Selectionner les valeurs à utiliser pour notre analyse
X = jointure.values
#Centrage / réduction des données pour que nos données puissent prendre la même importance
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)


# # <a name="C17"> 2- Méthode de classification ascendante hiérarchique "CAH", avec un dendrogramme comme visualisation</a>

# Création d'une matrice des liens selon la méthode de Ward
# **Source**:
# * Les méthodes de **Linkage** 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# * **CAH**
# https://www.xlstat.com/fr/solutions/fonctionnalites/classification-ascendante-hierarchique-cah#:~:text=classification%20ascendante%20hi%C3%A9rarchique-,La%20classification%20ascendante%20hi%C3%A9rarchique%20(CAH)%20est%20une%20m%C3%A9thode%20de%20classification,%C3%A0%20la%20nature%20des%20donn%C3%A9es.

# In[56]:


Z = linkage(X_scaled, method = 'ward', metric='euclidean')


# In[57]:


#Clustering hiérarchique: 
#Affichage d'un premier dendrogramme global
fig =plt.figure(figsize=(20,10))
sns.set_style('white')
plt.title('Clustering hiérarchique Dendrogramme', fontsize=20)
plt.ylabel('Distance')
dendrogram(Z, labels = jointure.index, leaf_font_size=10, color_threshold=12, orientation='top')
plt.xticks(rotation=80)
plt.show()


# * Le nombre de clusters sera de 4 pour cet ensemble de données.

# In[58]:


#Découpage du dendrogramme en 4 groupes pour avoir une première idée du partitionnement
fig = plt.figure(figsize=(6,6))
plt.title('Clustering hiérarchique Dendrogramme - 4 clusters', fontsize=20)
plt.xlabel('distance', fontsize=15)
dendrogram(Z, labels = jointure.index, p=4, truncate_mode='lastp', leaf_font_size=15, orientation='left')
plt.show()


# Nous optons pour 4 clusters qui se déclinent de la manière suivante :
# 
#     - Groupe 1 : 71 pays
#     - Groupe 2 : 23 pays
#     - Groupe 3 : 44 pays
#     - Groupe 4 : 31 pays

# # <a name="C18"> 3- La méthode des K-means</a>

# # <a name="C19"> 3-1- Méthode du coude</a>

# Elle est basée sur le fait que la somme de la variance intraclusters peut être réduite grâce à l'augmentation du nombre de clusters. Plus il est élevé, plus il permet d'extraire des groupes plus fins à partir de l'analyse d'objets de données qui ont plus de similarité entre eux. On utilise le point de retournement de la courbe de la somme des variances pour choisir le bon nombre de clusters.

# ![Capture%20d%E2%80%99e%CC%81cran%202023-02-03%20a%CC%80%2014.46.10.png](attachment:Capture%20d%E2%80%99e%CC%81cran%202023-02-03%20a%CC%80%2014.46.10.png)

# In[59]:


# Une liste vide pour enregistrer les inerties :  
intertia = [ ]

# Notre liste de nombres de clusters : 
k_range = range(1, 10)

# Pour chaque nombre de clusters : 
for k in k_range : 
    
    # On instancie un k-means pour k clusters
    kmeans = KMeans(n_clusters=k)
    
    # On entraine
    kmeans.fit(X_scaled)
    
    # On enregistre l'inertie obtenue : 
    intertia.append(kmeans.inertia_)
fig = plt.figure(figsize=(10,6))
plt.plot(k_range,intertia)
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele (Inertia)')
plt.grid()
plt.show()


# # <a name="C20"> 3-2- Affichage des clusters et centroïdes</a>

# In[60]:


# Affichage du nuage de points (individus) en cluster avec les centroïdes
fig = plt.figure(figsize=(8,4))
model = KMeans(n_clusters=4)
model.fit(X_scaled)
model.predict(X_scaled)
plt.scatter(X_scaled[:,0], X_scaled[:,1],c=model.predict(X_scaled))
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='^',c='c')
plt.grid()
plt.show()
# La somme des distances entre les points des clusters et centroïdes
print('Inertie totale :',model.inertia_)


# # <a name="C21"> 3-3- Coefficient de silhoutte</a>

# In[61]:


tab_silhouette =[]
k_range=range(2,10)
for k in k_range:
    model=KMeans(n_clusters=k)
    cluster_labels=model.fit_predict(X_scaled)
    tab_silhouette.append(silhouette_score(X_scaled,cluster_labels))

fig = plt.figure(figsize=(8,4))
plt.plot(k_range,tab_silhouette)
plt.xlabel('nombre de clusters')
plt.ylabel('Coefficient de silhouette')
plt.grid()
plt.show()
tab_silhouette


# ### Observations
# 
# En vu d'un partitionnement optimal nous avons couplé 2 méthodes :
# 
# * **Méthode du coude**
# 
# * **Coefficient de silhouette**
# 
# Avec la méthode du coude, on remarque que l’inertie stagne à partir de **4 clusters**.
# 
# Comme pour la méthode du coude cette fois ci nous affichons l'évolution du coefficient de silhouette en fonction du nombre de clusters :
# 
# Le nombre de 4 clusters donne bien le coefficient de silhouette le plus élevé
# 
# L'affichage du nuage de points avec les 4 clusters et leur centroîdes grâce à l'algorithme Kmeans :
# 
# * Le nuage de points est étalé
# 
# * Le nombre de clusters est optimal, et centroîdes bien distants

# # <a name="C22"> 4- Analyse des groupes</a>

# # <a name="C23"> 4-1- Découpage en classes - Matérialisation des groupes</a>

# In[62]:


#Identification des 4 groupes obtenus
groupes_cah = fcluster(Z, 4, criterion='maxclust')
#index triés des groupes
idg = np.argsort(groupes_cah)
#Affichage des pays selon leurs groupes
df = pd.DataFrame(jointure.index[idg], groupes_cah[idg]).reset_index()
df2 = df.rename(columns={'index':'Groupe'})
df2.head()


# In[63]:


#Intégration des références des groupes dans notre échantillon de départ représenté par le dataframe "dispoAlim"
#Jointure interne nécessaire pour parvenir à agréger nos données
df3 = pd.merge(jointure, df2, on='Zone')
df3.set_index('Zone', inplace=True)
df3.head()


# ### Moyenne des variables par groupe 

# In[64]:


# Moyenne globale
jointure.mean()


# In[65]:


#afficher les moyennes des variables de chaque groupe
groupe=(1,2,3,4)
for n in groupe:
    affi=df3.loc[df3['Groupe']==n].mean()
    print(affi)


# ## Les 4 clusters 

# In[66]:


# Cluster 1
cluster_1=df3.loc[df3['Groupe']==1]
cluster_1.shape


# In[67]:


cluster_1.head(31)


# In[68]:


# Cluster 2
cluster_2=df3.loc[df3['Groupe']==2]
cluster_2.shape


# In[69]:


cluster_2.head(44)


# In[70]:


# Cluster 3
cluster_3=df3.loc[df3['Groupe']==3]
cluster_3.shape


# In[71]:


cluster_3.head(23)


# In[72]:


# Cluster 4
cluster_4=df3.loc[df3['Groupe']==4]
cluster_4.shape


# In[73]:


cluster_4.head(50)


# # <a name="C24"> 4-2- Représentation de la distribution des variables par groupe en utilisant une boite à moustache</a>

# In[74]:


#Comparaison visuelle des groupes par Boxplot, en abscisse les numéros des groupes
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")
plt.subplot(221)
sns.boxplot(data=df3, x='Groupe', y='Croissance démographique (%)')
plt.subplot(222)
sns.boxplot(data=df3, x='Groupe', y='PIB (%)')
plt.subplot(223)
sns.boxplot(data=df3, x='Groupe', y='TAS (%)')
plt.subplot(224)
sns.boxplot(data=df3, x='Groupe', y='TDI (%)')
plt.savefig('Distribution des variables par groupe.jpg')


# In[75]:


plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")
plt.subplot(221)
sns.boxplot(data=df3, x='Groupe', y='Disponibilité alimentaire en quantité (kg/personne/an)')
plt.subplot(222)
sns.boxplot(data=df3, x='Groupe', y='Disponibilité alimentaire (Kcal/personne/jour)')
plt.subplot(223)
sns.boxplot(data=df3, x='Groupe', y='Disponibilité de protéines en quantité (g/personne/jour)')
plt.savefig('Distribution des variables par groupe_suite.jpg')


# ### caractéristiques de chaque groupe

# #### Groupe 1
# Ce groupe inclus essentiellement des pays avec :
# 
# * Un taux de dépendance à l'importation des plus élevé
# * Un taux d'auto-suffisance des plus faibles
# * Une croissance démographique des plus élevée
# * Une disponibilité des plus faible
# * Un PIB des plus élevé

# #### Groupe 2
# Ce groupe inclus essentiellement des pays avec :
# 
# * Un taux de dépendance à l'importation des plus faible
# * Un taux d'auto-suffisance des plus élevé
# * Une croissance démographique élevée
# * Une disponibilité très faible
# * Un PIB élevé

# #### Groupe 3
# 
# Ce groupe inclus essentiellement des pays avec :
# 
# * Un taux de dépendance à l'importation élevé
# * Un taux d'auto-suffisance très faible
# * Une croissance démographique faible
# * Une disponibilité très élevée
# * Un PIB des plus faible

# #### Groupe 4
# 
# Ce groupe inclus essentiellement des pays avec :
# 
# * Un taux de dépendance à l'importation des plus faible
# * Un taux d'auto-suffisance des plus élevé
# * Une croissance démographique faible
# * Une disponibilité élevée
# * Un PIB élevé

# ### Observations
# 
# Un groupe présente des caractéristiques intéressantes pour notre objectif d'exportation :
# 
# Le **groupe 1**
# Au vu des caractéristiques de chaque groupe la combinaison la plus favorable à la selection du meilleur groupe en terme de besoin de viande de volaille met en avant le groupe numéro 1 :
# 
# * Disponibilité alimentaire en quantité (kg/personne/an) :      **12.099032**
# * Disponibilité alimentaire (Kcal/personne/jour)            :   **41.677419**
# * Disponibilité de protéines en quantité (g/personne/jour) :     **4.209355**
# * TAS (%)                                                    :  **21.248387**
# * TDI (%)                                                   :  **103.223226**
# * Croissance démographique (%)                                :  **1.894516**
# * PIB (%)                                                     :  **7.230000**
# 
# 
# * Le taux de dépendance à l'importation est très élevé contrairement au taux d'auto-suffisance qui est très bas.
# 
# * Le PIB est le plus élevé
# 
# * Une croissance démographique importante
# 
# * Les disponibilités alimentaires sont égalements relativement basses.

# # <a name="C25"> 4-3- Croisement entre les différents clusters de pays avec les différentes variables</a>

# In[76]:


#heatmap avec les croisements entre les clusters de pays et les différentes variables
plt.figure(figsize=(7, 7))
sns.clustermap(jointure, cmap = 'viridis',method = 'ward',metric = 'euclidean',standard_scale =1,figsize = (18, 18))
plt.title('CLUSTER')


# ### Observations
# 
# La heatmap met bien en évidence la combinaison du cluster 1 avec les variables (indicateurs) :
# 
#  - TDI élevé
#  - Croissance démographique élevée
#  - TAS faibe
#  - PIB élevé
#  - Disponibilités faibles
#  
# Le groupe numéro 1 semble bien être le cluster idéal. Nous continuons notre analyse pour confirmer ces observations

# # <a name="C26"> 4-4- Corrélations entre les variables dans chaque groupe</a>

# Nous recherchons un cluster qui corréspond le plus au critères ci dessous :
# 
# * TDI négativement corrélé au TAS, aux disponibilités et population totale
# 
# 
# Cela corrésponderait aux pays qui ont encore des besoins en viande de volailles

# In[77]:


# heatmap de corrélations entre les différentes variables dans chaque cluster
groupe=(cluster_1,cluster_2,cluster_3,cluster_4)
for n in groupe:
    sns.heatmap(n.corr(),cmap='viridis')  
    plt.show()


# ### Observations
# 
# La heatmap du cluster numéro 1 confirme bien notre choix

# # <a name="C27"> 5- Analyse des composantes principales "ACP"</a>

# In[78]:


# Nous allons travailler que sur les 5 premières composantes :
n_components = 5

# Calcul des composantes principales (On instancie notre ACP :)
pca = PCA(n_components=n_components)
# On l'entraine sur les données scalées :
pca.fit(X_scaled)
#Intéressons nous maintenant à la variance captée par chaque nouvelle composante. 
#Grace à scikit-learn on peut utiliser l'attribut explained_variance_ratio_ :
pca.explained_variance_ratio_
#Enregistrons cela dans une variable :
scree = (pca.explained_variance_ratio_*100).round(2)
print(scree)
scree_cum = scree.cumsum().round()
print(scree_cum)
# Définisons ensuite une variable avec la liste de nos composantes :
x_list = range(1, n_components+1)
list(x_list)


# In[79]:


#On peut enfin l'afficher de façon graphique :
plt.figure(figsize=(8,6))
plt.bar(x_list, scree)
plt.plot(x_list, scree_cum,c="red",marker='o')
plt.xlabel("rang de l'axe d'inertie")
plt.ylabel("pourcentage d'inertie")
plt.title("Eboulis des valeurs propres")
plt.show(block=False)


# ### Observations
# 
# * On a en bleu la variance de chaque nouvelle composante, et en rouge la variance cumulée.
# 
# Nous avons dans notre cas l'inertie totale répartie inéquitablement sur 5 axes
# 
# - Axe 1 : 45.87 % de l'inertie totale
# - Axe 2 : 22.53 % de l'inertie totale
# - Axe 3 : 14.06 % de l'inertie totale
# - Axe 4 : 10.62 % de l'inertie totale
# - Axe 5 : 6.43 % de l'inertie totale
# 
# 
# * On voit ici que plus de 80% de la variance est comprise dans les 3 premières composantes, et plus de 90% dans les 4 premières.

# ## Components

# In[80]:


#La formule de ce calcul nous est donnée par l'attribut components_. Cette variable est généralement nommée pcs :
pcs = pca.components_
# Affichons la même chose mais version pandas :
pcs = pd.DataFrame(pcs)


# In[81]:


# choix du nombre de composantes à calculer
n_comp = 5
# selection des colonnes à prendre en compte dans l'ACP
features = jointure.columns
# Cercle des corrélations


# # <a name="C28"> 6- Cercles des corrélations</a>

# In[82]:


# Définissons nos axes x et y. Nous allons utiliser les 2 premières composantes. Comme - en code - on commence à compter à partir de 0, cela nous donne :
x_y = (0,1)
x_y
#On peut en faire une fonction :
def correlation_graph(pca, 
                      x_y, 
                      features) : 
    """Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i])
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # J'ai copié collé le code sans le lire
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)
# cercle des corrélations (F1 et F2)
correlation_graph(pca, (0,1), features)


# In[83]:


#Essayons pour F3 et F4 :
correlation_graph(pca, (2,3), features)


# ## Corrélation des variables avec les composantes principales

# In[84]:


# création de la matrice de corrélation
t=pca.components_[:]
df_corr_comp = pd.DataFrame(t,index = ['F1', 'F2', 'F3','F4','F5'],columns=jointure.columns)
df_corr_comp.head()


# ### OBSERVATIONS
# 
# **F1 :** 
# 
# - Variables corrélées positivement: Toutes les disponibiltées sont corrélées (0.55)
# 
# - On peut dire que l'axe F1 représente les disponibilitées
# 
# **F2 :**
# 
# - variables corrélées positivement : Le TDI est fortement corrélé (0.58)
# - variables corrélées négativement : Le TAS est fortement corrélé (- 0,68)
# 
# * On peut dire que les pays avec un fort TDI ont une tendance positive en terme de croissance démographique et un faible TAS
# 
# **F3 :**
# 
# - variables corrélées négativement : le PIB est très fortement corrélé (- 0,92)
# 
# **F4 :**
# 
# - variables corrélées positivement : Le TDI est fortement corrélé (0.54)
# - variables corrélées négativement : La croissance démographique est fortement corrélé (- 0.71)
# 
# * On peut dire que sur cet axe les pays avec un fort TDI ont également une tendance positive en terme de TAS et une croissance démographique faible

# # <a name="C29"> 7- Projections des individus</a>

# * Nous utiliserons 4 composantes principales pour projeter les individus.

# In[85]:


# Travaillons maintenant sur la projection de nos dimensions. 
# Tout d'abord calculons les coordonnées de nos individus dans le nouvel espace :
X_projected = pca.transform(X_scaled)
X_projected[:5]


# In[86]:


# Rappelons que :
x_y
def display_factorial_planes(   X_projected, 
                                x_y, 
                                pca=None, 
                                labels = None,
                                clusters=None, 
                                alpha=1,
                                figsize=[10,8], 
                                marker="." ):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # on définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters
 
    # Les points    
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha, 
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c,palette=['green','orange','brown','red'],)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) : 
        # j'ai copié collé la fonction sans la lire
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()


# In[87]:


# Essayons la version simple avec F1 et F2, les couleurs correspondent au groupe :
x_y = [0,1]
display_factorial_planes(X_projected, x_y,pca, clusters=df3["Groupe"],alpha = 1)


# In[88]:


# Essayons la version simple avec F3 et F4, les couleurs correspondent au groupe :
x_y = [2,3]
display_factorial_planes(X_projected, x_y,pca, clusters=df3["Groupe"],alpha = 1)


# ### Observations
# 
# Nous observons sur le plan 1, le groupe 1 qui se présente bien sur l'axe F1 en partie positive du TDI et négative de l'axe F1 (disponibilités).
# 
# Ce cluster présente bien les critères suivants :
# 
# * TDI élévé
# * TAS faible
# * Croissance démographique relativement élevée
# * Des disponibilités très faibles

# In[89]:


# Inserer la colonne 'population totale' et afficher les résultats par disponibilité en kcal (affichage ascendant)
population_totale=df_pop_nv.loc[:,[2017]]
population_totale.rename(columns ={2017: "Population totale"}, inplace= True)
cluster_final =cluster_1.merge(population_totale,on='Zone',how='inner')
groupe_candidat=cluster_final.loc[:,cluster_final.columns != 'Groupe']
groupe_candidat.sort_values(by = ['Disponibilité alimentaire (Kcal/personne/jour)'], ascending = True).head()


# # <a name="C30"> 8- Exploration du cluster selectionné</a>

# In[91]:


Y= cluster_1.values
# centrer et réduire les données
std_scale2 = preprocessing.StandardScaler().fit(Y)
X_scaled2 = std_scale2.transform(Y)
# création d'une Matrice des liens selon la Méthode de Ward
Z2 = linkage(X_scaled2, method = 'ward', metric='euclidean')
inertia2 = []
k_range2=range(1,10)
for k2 in k_range2:
    kmeans = KMeans(n_clusters=k2)
    kmeans.fit(Z2)
    inertia2.append(kmeans.inertia_)
plt.figure(figsize=(8,6))
plt.plot(k_range2,inertia2 )
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele (Inertia)');


# In[92]:


Z2 = linkage(X_scaled2, method = 'ward', metric='euclidean')
#Clustering hiérarchique: 
#Affichage d'un premier dendrogramme global
fig =plt.figure(figsize=(20,10))
sns.set_style('white')
plt.title('Clustering hiérarchique Dendrogramme', fontsize=20)
plt.ylabel('Distance')
dendrogram(Z2, labels = cluster_1.index, leaf_font_size=10, color_threshold=12, orientation='top')
plt.xticks(rotation=80)
plt.show()


# ### Affichage des clusters avec la méthode KMeans

# In[93]:


# Affichage du nuage de points (individus) en cluster avec les centoîdes
fig = plt.figure(figsize=(8,4))
model3 = KMeans(n_clusters=4)
model3.fit(Z2)
model3.predict(Z2)
plt.scatter(Z2[:,0], Z2[:,1],c=model3.predict(Z2))
plt.scatter(model3.cluster_centers_[:,0], model3.cluster_centers_[:,1], marker='^',c='c')
plt.grid()
plt.show()
print('Inertie totale :',model3.inertia_)


# In[94]:


#Identification des 3 groupes obtenus
groupes = fcluster(Z2, 4, criterion='maxclust')
#index triés des groupes
idg2 = np.argsort(groupes)
#Affichage des pays selon leurs groupes
df_groupe = pd.DataFrame(cluster_1.index[idg2], groupes[idg2]).reset_index()
df_groupe2 = df_groupe.rename(columns={'index':'Sous_Groupes'})
df_groupe2['Sous_Groupes'].unique()


# In[95]:


#Intégration des références des groupes dans notre échantillon de départ représenté par le dataframe
#Jointure interne nécessaire pour parvenir à agréger nos données
df_gr = pd.merge(cluster_1, df_groupe2, on='Zone')
df_gr.set_index('Zone', inplace=True)
df_groupe3 =df_gr.merge(population_totale,on='Zone',how='inner')
df_groupe3.head()


# In[96]:


#Comparaison visuelle des groupes par Boxplot, en abscisse les numéros des groupes
plt.figure(figsize=(20, 20))
sns.set(style="whitegrid")
plt.subplot(221)
sns.boxplot(data=df_groupe3, x='Sous_Groupes', y='Croissance démographique (%)')
plt.subplot(222)
sns.boxplot(data=df_groupe3, x='Sous_Groupes', y='Disponibilité alimentaire en quantité (kg/personne/an)')
plt.subplot(223)
sns.boxplot(data=df_groupe3, x='Sous_Groupes', y='TAS (%)')
plt.subplot(224)
sns.boxplot(data=df_groupe3, x='Sous_Groupes', y='TDI (%)')


# In[97]:


plt.figure(figsize=(20, 20))
sns.set(style="whitegrid")
plt.subplot(221)
sns.boxplot(data=df_groupe3, x='Sous_Groupes', y='Disponibilité alimentaire (Kcal/personne/jour)')
plt.subplot(222)
sns.boxplot(data=df_groupe3, x='Sous_Groupes', y='Disponibilité de protéines en quantité (g/personne/jour)')
plt.subplot(223)
sns.boxplot(data=df_groupe3, x='Sous_Groupes', y='PIB (%)')
plt.subplot(224)
sns.boxplot(data=df_groupe3, x='Sous_Groupes', y='Population totale')


# In[98]:


# Cluster 1_1
cluster_1_1=df_groupe3.loc[df_groupe3['Sous_Groupes']==1]
cluster_1_1.head()


# In[99]:


# Cluster 1_2
cluster_1_2=df_groupe3.loc[df_groupe3['Sous_Groupes']==2]
cluster_1_2.head()


# In[100]:


# Cluster 1_3
cluster_1_3=df_groupe3.loc[df_groupe3['Sous_Groupes']==3]
cluster_1_3.head()


# In[101]:


# Cluster 1_4
cluster_1_4=df_groupe3.loc[df_groupe3['Sous_Groupes']==4]
cluster_1_4.head()


# In[102]:


#afficher les moyennes des variables de chaque sous - groupes
groupe=(1,2,3,4)
for n in groupe:
    moy=df_groupe3.loc[df_groupe3['Sous_Groupes']==n].mean()
    print('Groupe',n,moy)


# ### Observations
# Comme pour l'analyse des groupes, certains sous-groupes présentent des caractéristiques plus favorables à notre objectif.
# 
# Le groupe 4 présente les critères :
# 
# * TDI très élevé
# * TAS très faible
# * Disponiblités faibles
# * PIB élevé
# 
# 
# * On peut constater que ces pays présentent des dispnibilités faibles alors que le TDI est élevé
# 
# * Ces pays peuvent être une destination pertinente pour l'exportation de viande de volaille

# In[103]:


cluster_1_4.sort_values(by = ['TAS (%)'], ascending = True).head(10)


# # <a name="C31"> Conclusion</a>

# Le groupe de pays qui correspond au critères de selection en terme de besoins en viande de volaille est le groupe 1. De ce groupe nous avons selectionné les pays qui correspondent le mieux au profil recherché.
# 
# Nous optons pour le sous-groupe 4
# 
# * Mongolie
# * Tadjikistan
# * Haïti
# * Lesotho
# * Angola
# * Kirghizistan
# * Mauritanie
# * Guinée
# * Ghana
# * Libéria
# 
# 
# Pour tous ces pays le taux de dépendance à l'importation est élevé et inversement le taux d'autosuffisance est faible.
# 
# Les pays ayants les plus faibles disponibilités alors qu'ils sont très dépendants à l'importation pourraient correspondre tout à fait à notre besoin.
# 
# Cette liste sera affinée avec les équipes métiers

# In[ ]:




