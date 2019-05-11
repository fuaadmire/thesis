import pandas as pd
import numpy as np
import os
import re

reader = pd.read_csv('data/news_cleaned_2018_02_13.csv', chunksize=1)
with open('FakeNewsCorpus_fake.txt', 'w+') as a, open('FakeNewsCorpus_true.txt', 'w+') as b:
    for chunk in reader:
        if chunk.type.iloc[0]=="fake":
            text = chunk.content.iloc[0].replace("\n", " ")
            text = re.sub(r"https?:\/\/.+" , " ",text)
            a.write(text)
            a.write("\n")
        elif chunk.type.iloc[0]=="reliable":
            text = chunk.content.iloc[0].replace("\n", " ")
            text = re.sub(r"https?:\/\/.+" , " ",text)
            b.write(text)
            b.write("\n")
