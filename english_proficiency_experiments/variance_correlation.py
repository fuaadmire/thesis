import sys
from scipy import stats
import numpy as np

Grades=[l.strip().split()[0] for l in open("data/proficiency_entire_docs.txt",'r').readlines()]
M = np.genfromtxt("data/old/student_preds_softmax_entire_docs.txt", delimiter=' ',skip_header=0)

A,B=[],[]
M_normed = M
for i in range(len(Grades)):
   A.append(M[i,0])
   B.append(float(Grades[i]))

even=True
A_,B_=[],[]
for (a,b) in zip(A,B):
   if even:
       (prev_a,prev_b)=(a,b)
       even=False
   else:
       if b in range(20,40):
           A_.append(abs(a-prev_a))
           B_.append(b)
       even=True
#print(A_)
#print(B_)
print("diff:"+str(stats.spearmanr(A_,B_)))
