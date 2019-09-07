import re
import pandas as pd
import sys


# first save the jsonl file as a txt file.
# cat file > newfile.txt

txtfile = sys.argv[1]

text_re = "'text': \['([^\]]*)" # make sure to get the whole text be ending at ], and not ' which may be used in the text.
#title_re = "'title': '([^']*)"
gender_re = "'gender': '([^']+)"
location_re = "'location': '([^']*)"



reviews = []
genders = []
locations = []

with open(txtfile, "r") as f:
    for line in f:
        text = re.findall(text_re, line)
        raw = " ".join(text) # getting out of list
        raw = raw[:len(raw)-1] # remove the ' at the end
        reviews.append(raw)
        gender = re.findall(gender_re, line)
        gender = " ".join(gender)
        genders.append(gender)
        loc = re.findall(location_re, line)
        loc = " ".join(loc)
        locations.append(loc)

df = pd.DataFrame(list(zip(locations, genders, reviews)), columns=['location','gender','review']).to_csv('TP_'+txtfile+'.csv')

# inspect missing values
# len(test)-pd.isnull(test["review"]).sum() # number of actual reviews
# df = test[pd.notnull(test["review"])] # take only data where review is not missing
# then replace nan values in gender with some unknown character
