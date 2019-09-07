import codecs
import ast
import sys
import pandas as pd
#parse jsonl files from trustpilot
#jsonl files contain one json object per line
#first I replaced all " in the text with \" using sed
#then I replaced all ' with " using sed
#According to json standards, use double quotes.

jfile = sys.argv[1]
print(jfile)
id = sys.argv[2]

count = 0
#directory = "~/Documents/KU/Speciale/thesis/english_proficiency_experiments/"
reviews = []
for line in codecs.open(jfile, 'r', encoding='utf-8'):
   try:
       l = ast.literal_eval(line.encode('utf-8'))
       l = line.encode('utf-8')
       reviews.append(l)
   except (SyntaxError, ValueError) as e:
       count+=1
       pass

print(count)




from pandas.io.json import json_normalize
alls = pd.DataFrame()
i=0
for r in reviews:
   df = pd.DataFrame.from_dict(json_normalize(data = r, record_path=['reviews'], meta=['location', 'gender'], errors='ignore'), orient='columns')

   geo = r.get('geocodes') # for some users geocodes (longitude and latitude are provided)
   if geo:
       geodata = json_normalize(data = r, record_path=['geocodes'], errors='ignore')
       df = pd.concat([df, geodata], axis=1)
       df[[ u'lat', u'lng']] = df[[ u'lat', u'lng']].ffill() # I don't use geonames and getcodes, because only a subset has that
   alls = pd.concat([df, alls], axis=0, ignore_index=True, sort=False)
   i+=1
   if i%10000==0:
       print(i, 'users done')

alls.to_csv(id+'Trustpilot_DK.csv', index=None, encoding='utf-8', sep='\t')
