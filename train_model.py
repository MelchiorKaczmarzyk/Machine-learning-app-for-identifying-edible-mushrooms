import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pickle

# Wczytaj danepip 
shrooms = pd.read_csv('mushroom.csv', usecols=['class','cap-diameter', 'cap-color', 'gill-color', 'stem-height', 'stem-width', 
                                                                                                      'stem-color', 'cap-shape', 'does-bruise-or-bleed', 'habitat', 'season'])
categoricals = ['class','cap-shape', 'cap-color', 'stem-color', 'gill-color','does-bruise-or-bleed', 'habitat', 'season']
df = pd.get_dummies(shrooms, columns = categoricals)
shroom_features = ['cap-diameter', 'stem-height', 'stem-width', 'cap-shape_b',
       'cap-shape_c', 'cap-shape_f', 'cap-shape_o', 'cap-shape_p',
       'cap-shape_s', 'cap-shape_x', 'cap-color_b', 'cap-color_e',
       'cap-color_g', 'cap-color_k', 'cap-color_l', 'cap-color_n',
       'cap-color_o', 'cap-color_p', 'cap-color_r', 'cap-color_u',
       'cap-color_w', 'cap-color_y', 'stem-color_b', 'stem-color_e',
       'stem-color_f', 'stem-color_g', 'stem-color_k', 'stem-color_l',
       'stem-color_n', 'stem-color_o', 'stem-color_p', 'stem-color_r',
       'stem-color_u', 'stem-color_w', 'stem-color_y', 'gill-color_b',
       'gill-color_e', 'gill-color_f', 'gill-color_g', 'gill-color_k',
       'gill-color_n', 'gill-color_o', 'gill-color_p', 'gill-color_r',
       'gill-color_u', 'gill-color_w', 'gill-color_y',
       'does-bruise-or-bleed_f', 'does-bruise-or-bleed_t', 'habitat_d',
       'habitat_g', 'habitat_h', 'habitat_l', 'habitat_m', 'habitat_p',
       'habitat_u', 'habitat_w', 'season_a', 'season_s', 'season_u',
       'season_w']
df = df.drop(columns=['class_p'])
X = df[shroom_features]
startTo = 50999
endFrom = 11000
Xtrain = X.iloc[:startTo ,:]
Xtest = X.iloc[endFrom:,:]
dfTrain = df.iloc[:startTo,:]
dfTest = df.iloc[endFrom:,:]
y = df['class_e']
ytrain = dfTrain['class_e']
ytest = dfTest['class_e']
shroom_model = DecisionTreeRegressor(random_state=1)
shroom_model.fit(Xtrain, ytrain)

# Zapisz model do pliku
modelFileName = 'model_file'
pickle.dump(shroom_model, open(modelFileName, 'wb'))

predictions = (shroom_model.predict(Xtest))
score = metrics.accuracy_score(ytest, predictions)
#print(mean_absolute_error(ytest, predictions))
print(score)