from dataset import ToSpectrograms, CusttomDataSnor
import pandas as pd  
import torch 
from sklearn.model_selection import train_test_split
from glob import glob 
import matplotlib.pyplot as plt

df = pd.DataFrame()
df["Link"] = glob("Snoring Dataset/*/*.wav")
df["Label"] = df["Link"].apply(lambda x: x.split("/")[-2])

df_train, df_test = train_test_split(df,
                        random_state=104,
                        test_size=0.2,
                        shuffle=True)

data = CusttomDataSnor(df_train)