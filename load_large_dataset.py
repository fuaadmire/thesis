import pandas as pd
import numpy as np
import os
from preprocessing_for_logreg import make_txt_file_from_pandas_textcolumn



df = pd.read_csv("data/news_cleaned_2018_02_13.csv")
fake = df[df.tags==fake]
true = df[df.tags==reliable]

fake_content = fake.content
true_content = true.content

make_txt_file_from_pandas_textcolumn(fake_content[:900000], "FakeNewsCorpus_fake_content.txt")
make_txt_file_from_pandas_textcolumn(true_content[:900000], "FakeNewsCorpus_true_content.txt")
