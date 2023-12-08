import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Do not run unless you comment out the last two '.to_csv' lines or it will overwrite the files


#read in dataset
df = pd.read_csv("train_v2_drcat_02_raw.csv")
print(df.head())

#get titles of prompts
prompt_name_series = df["prompt_name"]
prompt_name_set = set(prompt_name_series)

#initialize training and test sets
train = pd.DataFrame()
test = pd.DataFrame()

#look at each prompt
for pn in prompt_name_set:

    #split into human- and ai-generated texts
    human_df = df[(df["label"] == 0) & (df["prompt_name"] == pn)]
    ai_df = df[(df["label"] == 1) & (df["prompt_name"] == pn)]

    #see unbalanced numbers, note the smaller size
    print(pn,human_df.shape[0],ai_df.shape[0])
    smaller_size = min(human_df.shape[0],ai_df.shape[0])
    
    #cut down the size of the larger set
    undersampled_human_df = human_df.sample(n=smaller_size)
    undersampled_ai_df = ai_df.sample(n=smaller_size)
    
    #split into train and test sets with equal proportion of human- and ai-generated texts
    train_human, test_human = train_test_split(undersampled_human_df, test_size=0.2)
    train_ai, test_ai = train_test_split(undersampled_ai_df,test_size=0.2)
    train_pn = pd.concat([train_human,train_ai])
    test_pn = pd.concat([test_human,test_ai])

    #merge together with output train and test sets
    train = pd.concat([train,train_pn])
    test = pd.concat([test,test_pn])

print("Merged Train Set", train.shape[0])
print("Merged Test Set", test.shape[0])

#write to files
train.to_csv("train_preprocessed.csv")
test.to_csv("test_preprocessed.csv")