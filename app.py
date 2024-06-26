import pickle
import numpy as np
import pandas as pd

def ask_question(question, valid_options=None, is_float=False):
    while True:
        answer = input(question)
        if is_float:
            try:
                return float(answer)
            except ValueError:
                print("Proszę podać liczbę.")
        elif valid_options and answer not in valid_options:
            print(f"Proszę podać jedną z następujących wartości:\n{', '.join(valid_options)}.")
        else:
            return answer

features = ['cap-diameter', 'stem-height', 'stem-width', 'cap-shape', 'cap-color', 
            'stem-color', 'gill-color', 'does-bruise-or-bleed', 'habitat', 'season']

categories = {
    'cap-shape': [('b', 'bell'), ('c', 'conical '), ('f', 'flat'), ('o', 'others'), ('p', 'spherical'), ('s', 'sunken'), ('x', 'convex')],
    'cap-color': [('b', 'buff'), ('e', 'red'), ('g', 'gray'), ('k', 'black'), ('l', 'blue'), ('n', 'brown'), ('o', 'orange'), ('p', 'pink'), ('r', 'green'), ('u', 'purple'), ('w', 'white'), ('y', 'yellow')],
    'stem-color': [('b', 'buff'), ('e', 'red'), ('f', 'none'), ('g', 'gray'), ('k', 'black'), ('l', 'blue'), ('n', 'brown'), ('o', 'orange'), ('p', 'pink'), ('r', 'green'), ('u', 'purple'), ('w', 'white'), ('y', 'yellow')],
    'gill-color': [('b', 'buff'), ('e', 'red'), ('f', 'none'), ('g', 'gray'), ('k', 'black'), ('n', 'brown'), ('o', 'orange'), ('p', 'pink'), ('r', 'green'), ('u', 'purple'), ('w', 'white'), ('y', 'yellow')],
    'does-bruise-or-bleed': [('f', 'false'), ('t', 'true')],
    'habitat': [('d', 'woods'), ('g', 'grasses'), ('h', 'heaths'), ('l', 'leaves'), ('m', 'meadows'), ('p', 'paths'), ('u', 'urban'), ('w', 'waste')],
    'season': [('a', 'autumn'), ('s', 'spring'), ('u', 'summer'), ('w', 'winter')]
}

def convert_answers(answers):
    data = {feature: [0] for feature in expected_features}
    for feature in features:
        if feature in categories:
            for category in categories[feature]:
                col_name = f'{feature}_{category[0]}'
                if answers[feature] == category[0]:
                    data[col_name] = [1]
                else:
                    if col_name not in data:
                        data[col_name] = [0]
        else:
            data[feature] = [float(answers[feature])]
    return pd.DataFrame(data)

modelFileName = 'model_file'
loaded_model = pickle.load(open(modelFileName, 'rb'))

answers = {}
for feature in features:
    if feature in categories:
        options = '\n'.join([f"{opt[0]} - {opt[1]}" for opt in categories[feature]])
        question = f"Podaj wartość dla {feature}:\n{options}\n"
        answers[feature] = ask_question(question, valid_options=[opt[0] for opt in categories[feature]])
    else:
        question = f"Podaj wartość dla {feature}: "
        answers[feature] = ask_question(question, is_float=True)

expected_features = ['cap-diameter', 'stem-height', 'stem-width'] + \
    [f'cap-shape_{cat[0]}' for cat in categories['cap-shape']] + \
    [f'cap-color_{cat[0]}' for cat in categories['cap-color']] + \
    [f'stem-color_{cat[0]}' for cat in categories['stem-color']] + \
    [f'gill-color_{cat[0]}' for cat in categories['gill-color']] + \
    [f'does-bruise-or-bleed_{cat[0]}' for cat in categories['does-bruise-or-bleed']] + \
    [f'habitat_{cat[0]}' for cat in categories['habitat']] + \
    [f'season_{cat[0]}' for cat in categories['season']]

input_data = convert_answers(answers)

for feature in expected_features:
    if feature not in input_data.columns:
        input_data[feature] = 0

prediction = loaded_model.predict(input_data)

if prediction[0] == 1:
    print("Grzyb jest jadalny.")
else:
    print("Grzyb jest trujący.")