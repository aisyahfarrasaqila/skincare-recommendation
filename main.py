import pandas as pd
import numpy as np
import pickle
import joblib

# cluster dictionary
mapping = {0:'cleanser', 1:'mask', 2:'spot treatment', 3:'moisturizer', 4:'essence', 5:'serum', 6:'face mist', 7:'lip care', 8:'eye treatment', 9:'sun screen', 10:'booster', 11:'face oil', 12:'others'}

def get_cluster(input_list):
    input_list = np.array(input_list).reshape(1,20)
    model_path = "kmeans.pkl"
    loaded_model = pickle.load(open(model_path,"rb"))
    result = loaded_model.predict(input_list)
    result = [mapping[i] for i in result]
    return result[0]

def get_examples():
    df = pd.read_csv('beautyhaulclean.csv')
    examples = df.head(5)['product_name'].to_numpy().tolist()
    return examples
