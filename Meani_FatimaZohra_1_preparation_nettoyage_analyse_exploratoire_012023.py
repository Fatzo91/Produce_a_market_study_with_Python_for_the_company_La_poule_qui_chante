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

# # Partie 1 : Préparation et nettoyage des données

# # <a name="C1"> 1- Objectif </a>

# * L'objectif sera de proposer une première analyse des groupements de pays que l’on peut cibler pour exporter nos poulets. 
# * Nous approfondirons ensuite l'étude de marché. 

# # <a name="C2"> 2- Demandes</a>

# * Tester la classification ascendante hiérarchique, avec un dendrogramme comme visualisation. 
# 
# * Utiliser la méthode des k-means, afin d’affiner l’analyse et comparer les résultats des deux méthodes de clustering (Analyser les centroïdes des classes). 
# 
# * Réaliser une ACP afin de visualiser les résultats de l'analyse, comprendre les groupes, les liens entre les variables, les liens entre les individus.

# # <a name="C3"> 3- Le choix des indicateurs</a>

# * La croissance démographique (2016-2017)
# * Les disponibilités
# * Taux de dépendance aux importations (TDI)
# * Taux d'auto-suffisance (TAS)
# * Le taux de croissance du produit intérieur brut (PIB%) (2016-2017)

# # <a name="C4"> 4- Importation des librairies</a>

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import cluster, metrics
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from matplotlib.collections import LineCollection
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# # <a name="C5"> 5- Importation des données</a>

# A importer, les fichiers à notre possession :
# 
# * Le fichier de population (2000-2018) , i.e. Population_2000_2018.csv
# * Le fichier de disponibilite alimentaire (2017), i.e. DisponibiliteAlimentaire_2017.csv
# * Le fichier de produit intérieur brut (2016-2017), i.e. PIB.csv

# Nous pouvons à présent charger les fichiers csv, dans un dataframe que nous nommerons résepectivement (df_pop), (df_dispoAlim, (PIB). Nous affichons ensuite les 5 premières lignes.

# In[2]:


df_pop = pd.read_csv("Population_2000_2018.csv", decimal=".", index_col=0)
df_pop.head()


# In[3]:


df_dispoAlim = pd.read_csv("DisponibiliteAlimentaire_2017.csv", decimal=".", index_col=0)
df_dispoAlim.head()


# In[4]:


PIB = pd.read_csv("PIB.csv", decimal=".", index_col=0)
PIB.head()


# # <a name="C6"> 6- Préparation des données</a>

# #### Dimension de dataframe

# In[5]:


df_pop.shape


# In[6]:


df_dispoAlim.shape


# In[7]:


PIB.shape


# #### Indications globales de dataframe

# In[8]:


df_pop.info()


# In[9]:


df_dispoAlim.info()


# In[10]:


PIB.info()


# #### Le % de valeurs manquantes par colonne

# In[11]:


df_pop.isna().mean()


# In[12]:


df_dispoAlim.isna().mean()


# In[13]:


PIB.isna().mean()


# #### Les doublons

# In[14]:


df_pop.duplicated().sum()


# In[15]:


df_dispoAlim.duplicated().sum()


# In[16]:


PIB.duplicated().sum()


# #### Des valeurs différentes par colonne

# In[17]:


df_pop.nunique()


# In[18]:


df_dispoAlim.nunique()


# In[19]:


PIB.nunique()


# #### La dispersion globale de nos données

# In[20]:


df_pop.describe()


# In[21]:


df_dispoAlim.describe()


# In[22]:


PIB.describe()


# # <a name="C7"> 7- Création de nouvelles variables pour notre analyse</a>

# # <a name="C5"> 7-1- Taux de croissance démographique sur la période 2016-2017 </a>

# La **croissance démographique** ou variation totale de population est la différence entre l’effectif d’une population à la fin et au début d’une période donnée.

# In[23]:


df_pop_nv= df_pop.pivot_table(index = 'Zone', columns = 'Année', values = 'Valeur')
df_pop_nv


# In[24]:


# Croissance démographique (2016-2017)
df_pop_nv['Croissance démographique (%)']=round(100*(df_pop_nv[2017]-df_pop_nv[2016]) /(df_pop_nv[2016]),2)
df_pop_nv.head()


# In[25]:


# Conserver uniquement la colonne croissance démographique pour notre analyse
pop=df_pop_nv.loc[:,['Croissance démographique (%)']]
pop.head()


# In[26]:


pop.shape


# In[27]:


# 2 valeurs nulles 
pop.isnull().sum()


# # <a name="C9"> 7-2- Taux de dépendance aux importations (TDI) </a>

# * Au cours de l'analyse de la situation alimentaire d'un pays, il importe de bien savoir quelle part les approvisionnements intérieurs disponibles provient des importations et quelle part provient de la production du pays lui-même.
# 
# * Plus le TDI est élevé plus la dépendance du pays à l'importation est forte
# 
# ##### Taux de dépendance aux importations (TDI) = (Importation ÷ Disponibilité intérieure) x 100
# 
# Avec: 
# ##### DISPONIBILITE INTERIEUR = Production + importations - exportations + variations des stocks 

# In[28]:


# Éliminer les colonnes que nous n'utilisons pas
df_dispoAlim = df_dispoAlim.drop(['Domaine', "Code zone","Code Élément","Code Produit","Code année",'Année','Unité','Symbole',"Description du Symbole"],axis=1).reset_index(drop=True)


# In[29]:


# Conserver uniquement les données concernant le produit 'Viande de Volailles'
df_dispoAlim = df_dispoAlim.loc[df_dispoAlim['Produit']==  'Viande de Volailles']
df_dispoAlim


# In[30]:


# On crée une liste des éléments
liste_éléments=["Disponibilité intérieure","Disponibilité alimentaire (Kcal/personne/jour)",'Disponibilité alimentaire en quantité (kg/personne/an)','Disponibilité de protéines en quantité (g/personne/jour)',"Production", "Importations - Quantité"]


# In[31]:


dispoAlim = df_dispoAlim[df_dispoAlim["Élément"].isin(liste_éléments)]


# In[32]:


dispoAlim.set_index('Zone',inplace=True)


# In[33]:


# Faire un pivot de la colonne Élément et vérifier le nombre de valeurs nulles par colonne
dispoAlim=dispoAlim.pivot_table(index='Zone', columns='Élément', values = 'Valeur')
dispoAlim.isnull().sum()


# In[34]:


# Remplacer les valeurs nulles des colonnes par leurs moyennes
dispoAlim=dispoAlim.fillna(dispoAlim.mean())


# In[35]:


# Calculer le TDI 
dispoAlim['TDI (%)']=round((dispoAlim['Importations - Quantité']/dispoAlim['Disponibilité intérieure'])*100,2)
dispoAlim.head()


# # <a name="C10"> 7-3- Taux de autosuffisance  </a>

# * Le taux d'autosuffisance exprime l'importance de la production, par rapport à la consommation intérieure.
# 
# * Plus le TAS est élevé plus le pays est auto-suffisant c'est-à-dire la production est forte également
# 
# ##### Taux d'auto-suffisance (TAS) = (Production ÷ Disponibilité intérieure) x 100

# In[36]:


# Calculer le TAS
dispoAlim['TAS (%)']=round((dispoAlim['Production']/dispoAlim['Disponibilité intérieure'])*100,2)
dispoAlim.head()


# In[37]:


# Dataset avec les colonnes à utiliser pour les jointures et en suite notre analyse
dispoAlim= dispoAlim.loc[:,['Disponibilité alimentaire en quantité (kg/personne/an)','Disponibilité alimentaire (Kcal/personne/jour)','Disponibilité de protéines en quantité (g/personne/jour)','TAS (%)','TDI (%)']]
dispoAlim.head()


# # <a name="C11"> 7-4-  Le taux de croissance du produit intérieur brut (PIB%) (2016-2017)</a>

# In[38]:


# Éliminer les colonnes que nous n'utilisons pas
PIB_nv = PIB.drop(['Domaine', "Code zone (M49)","Code Élément","Élément","Code Produit","Produit","Code année",'Unité','Symbole',"Description du Symbole","Note"],axis=1).reset_index(drop=True)


# In[39]:


# Calculer une nouvelle variable pour notre analyse : le taux de croissance du produit intérieur brut en % entre 2016 et 2017
df_PIB= PIB_nv.pivot_table(index = 'Zone', columns = 'Année', values = 'Valeur')
df_PIB


# In[40]:


# Le taux de croissance du produit intérieur brut (PIB%) (2016-2017)
df_PIB['PIB (%)']=round(100*(df_PIB[2017]-df_PIB[2016]) /(df_PIB[2016]),2)
df_PIB.head()


# In[41]:


# Conserver uniquement la colonne PIB (%) pour notre analyse
df_PIB=df_PIB.loc[:,['PIB (%)']]
df_PIB.head()


# In[42]:


# Taille 
df_PIB.shape


# In[43]:


# Valeurs manquantes
df_PIB.isnull().sum()


# # <a name="C12"> 8- Jointure</a>

# In[44]:


# Effectuer la prmère jointure entre le dataset résulant de population avec celui des disponibilités alimentaires
jointure1=pd.merge(dispoAlim,pop,on='Zone',how='inner')
jointure1.head()


# In[45]:


# Taille
jointure1.shape


# In[46]:


# Effectuer la deuxième jointure avec le dataset PIB  
jointure_finale =pd.merge(jointure1,df_PIB,on='Zone',how='inner')
jointure_finale


# In[47]:


# Aucune valeur nulle
jointure_finale.isnull().sum()


# # Partie 2: Analyse exploratoire des données

# # <a name="C13"> 1- Corrélation entre les différentes variables</a>

# On affiche la corrélation entre les différentes variables avec une "**Heat Map**"
# 
# **Source**: https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

# In[48]:


# Increase the size of the heatmap.
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(jointure_finale.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Corrélation entre les différentes variables', fontdict={'fontsize':12}, pad=12);


# * Le taux de dépendance à l'importation (TDI) est **négativement (faiblement)** corrélé au PIB (%) 
# * Le taux de dépendance à l'importation (TDI) est **positivement (faiblement)** corrélé aux disponibilitées et TAS (%)

# # <a name="C14"> 2- Matrice de corrélation</a>

# In[49]:


# Afficher la matrice de corrélation
corr_df = jointure_finale.corr()
matrice_corr =pd.DataFrame(corr_df)
matrice_corr.head()


# # <a name="C15"> 3- Vérification de la distribution des variables de tous les individus</a>

# In[50]:


plt.figure(figsize=(12, 12))
sns.set(style="whitegrid")
plt.subplot(221)
sns.boxplot(data=jointure_finale,  y='Disponibilité alimentaire en quantité (kg/personne/an)')
plt.subplot(222)
sns.boxplot(data=jointure_finale,  y='Disponibilité alimentaire (Kcal/personne/jour)')
plt.subplot(223)
sns.boxplot(data=jointure_finale,  y='Disponibilité de protéines en quantité (g/personne/jour)')


# In[51]:


# Afficher des boxplot
plt.figure(figsize=(9, 8))
sns.set(style="whitegrid")
plt.subplot(221)
sns.boxplot(data=jointure_finale,y='Croissance démographique (%)')
plt.subplot(222)
sns.boxplot(data=jointure_finale, y='PIB (%)')
plt.subplot(223)
sns.boxplot(data=jointure_finale, y='TAS (%)')
plt.subplot(224)
sns.boxplot(data=jointure_finale,y='TDI (%)')


# #### Les valeurs extremes de l'auto-suffisance
# Nous observons des valeurs abbérantes pour toutes les variables mais nous optons pour l'exclusion uniquement des valeurs extrèmes des 2 variables TAS. les pays concernés :
# 
# * Djibouti : TAS (%) 24173.02
# * Maldives : TAS (%) 6043.25

# In[52]:


# Afficher les valeurs extremes de l'auto-suffisance
jointure_finale.sort_values(by = ['TAS (%)'], ascending = False).head()


# In[53]:


# Exclure les 2 pays qui représentent des valeurs extrèmes
jointure =jointure_finale.drop(['Djibouti','Maldives']) 
jointure.head()


# * Les pays dépendants à l'importation (TDI) sont ceux qui ont un taux d'auto-suffisance (TAS) le plus faible
# * Les pays avec un TDI important ont des disponibilités relativement faibles

# ### Les moyennes des variables de tous les individus 

# In[54]:


jointure.mean()

