# Files details

### DATA ANALYSIS :
    - discover_data.ipynb      : more relevant discovery and correlation between the data
    - discover_data_full.ipynb : discovery and correlation between the data - all plots

### LOCAL ML MODELS
    - local_model.ipynb   : dataset creation / model creation / training and experimentation
                                -> Train+test on a single point
                                -> Train on one point + test on another point (+ test for all points of the map)
                                -> Train + test on several points
                                -> Compare local an global approach
                                -> Random : quick tests with differents models (ConvLSTM, CNN).
                                   The database creation is not optimal as i just copy paste an old unoptimize code
    - find_area.ipynb          : Train + test on differents points
                                -> Determine differents areas with kmean classification between Xe-param time series for all points
                                -> Train + test ML model with one area



# Ideas of paths to explore
  
    
### Approches générales du problème
    - Approche locale :
        - 1 point = 1 modèle
        - 1 zone = 1 modèle
            - zone = points voisins (possibilité de faire des recouvrement entre zones pour avoir plusieurs prédictions d'un point)
            - zone = classifier des points (selon les corrélation de divers paramètres en ce point avec l'erreur ?) -> cf find_area.ipynb
            - ...
    - Modèle hybride approche locale/approche globale
        - Un modèle qui prend en entrée les résultats de l'approche globale / des PCA et les coordonnées du point
        - Modèle local pour certains points limties ?
        - ...
    
### Ajustements sur le modèle
    - Autres architectures de NN (CNN, ConvLSTM, Transformers, GRU ~ LSTM moins coûteux, transfert learning...)
    - Autres paramètres d'entrée :
        - Tester plusieurs covariables/forcings
        - Ajouter age glace / distance avec les côtes (plutôt dans le modèle par zones)
        - Donner différents points (contiguë, non, ?)
        - Donner plusieurs temps (nb de valeurs, passé/futur, quelle fréquence, ..?)
    - récursif ?