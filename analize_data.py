import pandas as pd

shrooms = pd.read_csv('mushroom.csv')

poisonous_shrooms = shrooms[shrooms['class'] == 'p']

print(poisonous_shrooms.describe(include='all'))