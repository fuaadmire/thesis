import numpy as np
import codecs
from bs4 import BeautifulSoup as bs
import bs4
from nltk.tokenize import sent_tokenize, word_tokenize
import glob
import string
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy import stats
from collections import Counter

def extract_error_c(elem):
    #if no embedded errors: return teacher and student version

    #try:
    #    typ = elem.NS['type']
        #print(typ)
    #except TypeError:
    #    pass

    teachersent =''
    studentsent =''

    if elem.find('c'):
        teachersent+=elem.c.getText()
    else:
        teachersent+=''

    if elem.find('i'):
        studentsent+=elem.i.getText()
    else:
        studentsent+=''
    return teachersent, studentsent

#Functions for parsing FCE corpus
def joinsents(old, new):
    new = new.strip()
    if len(old) == 0:
        joined =new
    elif old[-1] == ' ':
        if len(new) >0 and new[0] in string.punctuation:
            joined = old[:-1]+new
        else:
            joined = old+new
    else:
        if len(new) > 0 and new[0] in string.punctuation:
            joined = old+new
        else:
            joined = ' '.join([old, new])
    return(joined)

def teacherversion(tag):
    #print(tag.name)
    if tag.name == 'i':
        #print('its an i')
        return []
    elif tag.name == 'c':
        #print('its a c')
        return tag
    elif tag.parent.name == 'NS' and tag.parent['type'][0] in ['M'] or tag.parent['type'] != 'CE':
        print('missing')
        return tag

def studentversion(tag):
    #print('tag', tag)
    #print('tagparent', tag.parent)
    if tag.name == 'c':
        #print('its an i')
        return []
    elif tag.name == 'i':
        #print('its a c')
        return tag
    elif tag.parent.name == 'NS':
        print(tag)
        if tag.parent['type'][0] not in ['M'] or tag.parent['type'] == 'CE':
            print('missing')
            return tag
        else:
            return []


path = "data/proficiency/"
textout = codecs.open(path+'fce_text_entire_docs.txt', 'w', 'utf-8')
profout = codecs.open(path+'proficiency_entire_docs.txt', 'w', 'utf-8')
firstlangout = codecs.open(path+'firstlang_entire_docs.txt', 'w', 'utf-8')
docscoreout = codecs.open(path+'docscore_entire_docs.txt', 'w', 'utf-8')

#teachertextout = codecs.open(path+'fce_teacher', 'w', 'utf-8')
#teacherprofout = codecs.open(path+'proficiency_teacher', 'w', 'utf-8')
#teacherfirstlangout = codecs.open(path+'firstlang_teacher', 'w', 'utf-8')
langvec_t =[]
profvec_t =[]
docscorevec_t=[]
textvec_t = []
useridvec_t = []
ansvec_t = []


langvec_s =[]
profvec_s =[]
docscorevec_s=[]
textvec_s = []
useridvec_s = []
ansvec_s = []

for i, doc in enumerate(glob.glob(path+'fce-released-dataset/dataset/0100_2001_6/*xml')): #only test and validation
    print(i, doc)
    doc = codecs.open(doc, 'r', 'utf-8').read()
    soup = bs(doc, 'xml')

    firstlang = soup.language.string
    """if soup.age:  #Only around 30 out of 97 test documents have age
        agegroup=soup.age.string
    else:
        break
    """
    score=soup.score.string
    studentid=soup.head['sortkey']


    for ans in soup.find_all(['answer1', 'answer2']):
        docscore = ans.exam_score.string
        parag = ans.find_all('p')
        qnumber = ans.question_number.string
        student=""
        teacher=""
        for p in parag:
            for ch in p:
                if ch.find('NS') == -1 or type(ch.find('NS')) == int:#Then there are no errors in this bit
                    teacher = joinsents(teacher, ch)
                    student = joinsents(student, ch)
                else:
                    if ch.find('NS') and len(ch.find('NS')) > 0:
                        #print(ch)
                        for gr in ch.children:
                            if gr.find('NS') == -1:
                                #print(gr)
                                if not gr.find('i') or gr.find('c'):
                                    if gr.parent['type'] == 'CE' or gr.parent['type'][0] in ['R', 'U']:
                                        student = joinsents(student, gr)
                                    elif gr.parent['type'] in ['M']:
                                        teacher = joinsents(teacher, gr)
                                    else:
                                        teacher = joinsents(teacher, gr)
                                        student = joinsents(student, gr)
                                else:
                                    print('gr',gr)
                            else:
                                if gr.find('NS') and len(str(gr.find('NS'))) > 0:
                                    for ggc in gr.children:
                                        if ggc.find('NS') == -1:
                                            try:
                                                #print(ggc)
                                                if not ggc.find('i') or ggc.find('c'):
                                                    if ggc.parent.name=='NS' and ggc.parent['type'] == 'CE' or ggc.parent['type'][0] in ['R', 'U']:
                                                        student = joinsents(student, ggc)
                                                    elif ggc.parent['type'] in ['M']:
                                                        teacher = joinsents(teacher, ggc)
                                                    else:
                                                        teacher = joinsents(teacher, ggc)
                                                        student = joinsents(student, ggc)
                                                else:
                                                    print('ggc', ggc)
                                                    pass
                                            except KeyError:
                                                joinsents(teacher, ggc)
                                                joinsents(student, ggc)

                                        elif ggc.find('NS') and len(ggc.find('NS')) > 0:
                                            for gggc in ggc.children:
                                                if gggc.find('NS') == -1:
                                                    try:
                                                        #print(ggc)
                                                        if not gggc.find('i') or gggc.find('c'):
                                                            if gggc.parent.name=='NS' and gggc.parent['type'] == 'CE' or gggc.parent['type'][0] in ['R', 'U']:
                                                                student = joinsents(student, gggc)
                                                            elif gggc.parent['type'] in ['M']:
                                                                teacher = joinsents(teacher, gggc)
                                                            else:
                                                                teacher = joinsents(teacher, gggc)
                                                                student = joinsents(student, gggc)
                                                        else:
                                                            print('gggc', gggc)
                                                            pass
                                                    except KeyError:
                                                        joinsents(teacher, gggc)
                                                        joinsents(student, gggc)

                                        else:
                                            t = ggc.find_all(teacherversion)
                                            t = ' '.join([c.text for c in t])
                                            teacher = joinsents(teacher, t)
                                            s = ggc.find_all(studentversion)
                                            s = ' '.join([i.text for i in s])
                                            student = joinsents(student, s)
                                else: #if part of certain errors in parent, only one of versions
                                    if gr.parent['type'] == 'CE' or gr.parent['type'] in ['R', 'U']:
                                        s = gr.find_all(studentversion)
                                        s = ' '.join([i.text for i in s])
                                        student = joinsents(student, s)

                                    elif gr.parent['type'] in ['M']:
                                        t = gr.find_all(teacherversion)
                                        t = ' '.join([c.text for c in t])
                                        teacher = joinsents(teacher, t)
                                    else:
                                        s = gr.find_all(studentversion)
                                        s = ' '.join([i.text for i in s])
                                        student = joinsents(student, s)
                                        t = gr.find_all(teacherversion)
                                        t = ' '.join([c.text for c in t])
                                        teacher = joinsents(teacher, t)

                    else:
                        if ch.name == 'NS' and not ch.i and not ch.c:
                            if ch['type'] == 'CE' or ch['type'][0] in ['R', 'U']:
                                student = joinsents(student, ch.get_text())
                            if ch['type'][0] in ['M']:
                                teacher = joinsents(teacher, ch.get_text())
                        else:
                            t = ch.find_all(teacherversion)
                            t = ' '.join([c.text for c in t])
                            teacher = joinsents(teacher, t)
                            s = ch.find_all(studentversion)
                            s = ' '.join([i.text for i in s])
                            student = joinsents(student, s)
        if firstlang and score and len(student)>50:
            #studentsents = sent_tokenize(student)
            #teachersents = sent_tokenize(teacher)
            textout.write(student.strip('\n')+'\n')
            firstlangout.write(firstlang+'\t'+studentid+'\n')
            profout.write(score+'\t'+studentid+'\n')
            docscoreout.write(docscore+'\t'+studentid+'\n')
            # for sent in studentsents:
            #     langvec_s.append(firstlang)
            #     profvec_s.append(float(score))
            #     docscorevec_s.append(float(docscore.strip('T')))
            #     textvec_s.append(sent)
            #     useridvec_s.append(studentid)
            #     ansvec_s.append(qnumber)
            #     textout.write(sent.strip('\n')+'\n')
            #     firstlangout.write(firstlang+'\t'+studentid+'\n')
            #     profout.write(score+'\t'+studentid+'\n')
            #     docscoreout.write(docscore+'\t'+studentid+'\n')
            # for sent in teachersents:
            #     teachertextout.write(sent.strip('\n')+'\n')
            #     teacherfirstlangout.write(firstlang+'\t'+studentid+'\n')
            #     teacherprofout.write(score+'\t'+studentid+'\n')
            #     langvec_t.append(firstlang)
            #     profvec_t.append(float(score))
            #     docscorevec_t.append(float(docscore.strip('T')))
            #     textvec_t.append(sent)
            #     useridvec_t.append(studentid)
            #     ansvec_t.append(qnumber)

textout.close()
profout.close()
firstlangout.close()
docscoreout.close()
#teachertextout.close()
#teacherprofout.close()
#teacherfirstlangout.close()
