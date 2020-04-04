# coding=utf-8
import jieba
jieba.load_userdict("data/wholewords.txt")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math
import numpy as np
import time
from tqdm import tqdm
from sklearn.externals import joblib
from simhash import  *
import re
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

def clean(text):
    text=strQ2B(text)
    text=re.sub(r'\ue40c','',text)
    # text=re.sub('[:「」￥…，,(【嘻嘻】)【哈哈】;"”“+/—!. _ - % \[\]*◎《》、。]', '', text)  # 去特殊字符
    text = re.sub('[「」￥，,(【嘻嘻】)【哈哈】"”“+/—. _ - % \[\]*◎《》、]', '', text)  # 去特殊字符
    text=re.sub('(www\.(.*?)\.com)|(http://(.*?)\.com)','',text.lower()) #去URL
    text=re.sub('[a-zA-Z]+','',text) # 去英文
    text=re.sub('([\d]*年*[\d]*月*[\d]+日+)|([\d]+年+)|([\d]*年*[\d]+月+)','',text)#去日期
    text=re.sub('[\d]+','',text) #去数字
    return text

def CleanData(dataset, ad_words= ['借钱'] ):
    for i,document in enumerate(tqdm(list(dataset['正文'].values))):
        # 保留符号分割
        title = dataset['标题'].iloc[i]
        dataset['标题'].iloc[i] = title.split("|")[-1]
        # document = document[:min(250,len(document))]
        # document = clean(document)
        sentences = re.split('([。，\n ])',document)
        D=""
        for sentence in sentences:
            if_ads = False
            for regex in ad_words:  # 去除广告
                if re.search(regex,sentence) is not None:
                    if_ads = True #是广告
                    break
            if not if_ads:
                D = D + sentence
        dataset['正文'].iloc[i]=D
    return dataset

def PrepareData(excelFile = 'data/数据集.xlsx',Stop_path = 'data/stopwords.txt', max_n_samples = None, Maxcut=50, onlyload=False):
    # 读取数据集
    dataset = pd.read_excel(excelFile)
    # dataset = CleanData(dataset)
    # dataset.to_excel("data/数据集.xlsx", index=False)
    # print("数据清洗完成")
    if max_n_samples:
        dataset = dataset[:max_n_samples]

    with open(Stop_path, "r", encoding="utf-8") as f:
        Stopword = set(f.read().splitlines())
    if onlyload:
        return dataset, Stopword
    documents = list(dataset['正文'].values)
    print("读取数据完成")
    # 头Maxcut个字
    corpus_head = [del_stopword(e[:min(Maxcut,len(e))],Stopword) for e in documents]
    # 尾Maxcut个字 计算相似度
    corpus_tail = [del_stopword(e[-min(Maxcut, len(e)):], Stopword) for e in documents]
    # 标题
    titles = list(dataset['标题'].values)
    corpus_title = [del_stopword(e[:min(Maxcut, len(e))], Stopword) for e in titles]
    print("准备语料完成")

    return corpus_head, corpus_tail,corpus_title, documents, dataset, Stopword

# 去除停用词
def del_stopword(line,Stopword,ngram=False):
    line=list(jieba.cut(line))#分词
    new = [word  for word in line if word not in Stopword]
    if ngram :#返回2元语法
        N = len(line)
        for i, word_i in enumerate(line):
            for j in range(min(i+1,N-1),N):
                word_j = line[j]
                if word_i not in Stopword and word_j not in Stopword:
                    new.append(word_i+' '+word_j)
    return new # [w1,w2,...]

def quchong_bysklearn(dataset,Stopword, threhold = 0.5):
    ## 计算语料的TF-IDF矩阵
    documents = list(dataset['正文'].values)
    corpus = [" ".join(del_stopword(e[:min(50,len(e))],Stopword)) for e in documents]
    vectorizer = CountVectorizer()#ngram_range=(1,2)
    count = vectorizer.fit_transform(corpus)
    TFIDF= TfidfTransformer()
    tfidf_matrix = TFIDF.fit_transform(count)
    d_matrix = np.array(tfidf_matrix.toarray())
    ## 计算余弦相似度
    CosSim = cosine_similarity(d_matrix)
    n_samples, _ =  CosSim.shape
    repeat_index={}
    repeat_set = set()
    for i in range(n_samples):
        for j in range(i+1,n_samples):
            if j in repeat_set:
                continue
            if CosSim[i,j] > threhold:
                if i not in repeat_index:
                    repeat_index[i] = []
                repeat_index[i].append(j)
                repeat_set.add(j)
    for index in repeat_index:
        print("——————————————————————————")
        print("【原文】---ID",index,documents[index])
        for i in repeat_index[index]:
            print("【重复】---ID",i, documents[i])

##################### 说明 #####################

'''
通过TF-IDF可以检测重复文档，
但是16G内存下只能找出2W个文档中的重复文章，
下面尝试使用稀疏矩阵解决这个问题。
'''

def list2dic(l):
    tmp = {}
    for i in l:
        if i in tmp:
            tmp[i] += 1
        else:
            tmp[i] = 1
    return tmp

def calc_tfidf(corpus):
    tf, tmp = [], []
    for line in corpus:
        tf.append(list2dic(line))
    for i in tf:
        tmp.extend(i.keys())
    idf = list2dic(tmp)
    N = len(tf)
    for i in idf:
        idf[i] = math.log(N / (idf[i] + 1))
    for i in range(len(tf)):
        for word in tf[i]:
            tf[i][word] = tf[i][word] * idf[word]
    return tf

def cos_sim(x1, x2):
    if (not x1) | (not x2):
        return 0
    if (len(x1) == 0) | (len(x2) == 0):
        return 0
    fenzi, fenmu1, fenmu2 = 0, 0, 0
    for i in x1.keys():
        if i in x2:
            fenzi += x1[i] * x2[i]
        fenmu1 += x1[i] * x1[i]
    for i in x2.values():
        fenmu2 += i * i
    fenmu = math.sqrt(fenmu1) * math.sqrt(fenmu2)
    return fenzi / fenmu

def get_top(tfidf, top):
    # 根据tf-idf保留top的词
    return [dict(sorted(i.items(), key=lambda x: x[1], reverse=True)[:int(len(i) * top)]) for i in tfidf]

def compute_CosSim(corpus_head, corpus_tail,corpus_title, threhold = 0.5):
    # 计时
    start = time.time()
    d_matrix_1 = calc_tfidf(corpus_head)
    d_matrix_2 = calc_tfidf(corpus_tail)
    d_matrix_3 = calc_tfidf(corpus_title)

    print("计算TF-IDF矩阵完成")
    n_samples =  len(d_matrix_1)
    repeat_index={}
    repeat_set = set()

    print("计算相似度...\n")
    for i in tqdm(range(n_samples)):
        for j in range(i+1,n_samples):
            if j in repeat_set:
                continue
            if cos_sim(d_matrix_1[i], d_matrix_1[j]) > \
                    threhold or cos_sim(d_matrix_2[i], d_matrix_2[j]) > \
                    threhold or cos_sim(d_matrix_3[i], d_matrix_3[j]) > threhold :
                if i not in repeat_index:
                    repeat_index[i] = []
                repeat_index[i].append(j)
                repeat_set.add(j)
    end = time.time()
    print("耗时:{:.2f}秒".format((end-start)))
    return repeat_index, repeat_set

def print_quchong(documents, repeat_index):
    for index in repeat_index:
        print("——————————————————————————")
        print("【原文】---ID",index,documents[index])
        for i in repeat_index[index]:
            try:
                print("【重复】---ID",i, documents[i])
            except:
                continue
        if index > 1000:
            break

# 保存x
def save_result(repeat_index,repeat_set):

    joblib.dump(repeat_index, 'data/repeat_index.pkl')
    joblib.dump(repeat_set, 'data/repeat_set.pkl')

def load_result(indexpath='data/repeat_index.pkl', setpath='data/repeat_set.pkl'):
    repeat_index = joblib.load(indexpath)
    repeat_set = joblib.load(setpath)
    return repeat_index, repeat_set

def compute_simhash(corpus,threhold=10):
    start = time.time()
    n_samples =  len(corpus)
    repeat_index={}
    repeat_set = set()
    print("计算相似度...\n")
    for i in tqdm(range(n_samples)):
        simhash_i = simhash(tokens=corpus[i])
        for j in range(i+1,n_samples):
            if j in repeat_set:
                continue
            simhash_j = simhash(tokens=corpus[j])
            if simhash_i.hamming_distance(simhash_j.hash) < threhold:
                if i not in repeat_index:
                    repeat_index[i] = []
                repeat_index[i].append(j)
                repeat_set.add(j)

    end = time.time()
    print("耗时:{:.2f}秒".format((end-start)))
    return repeat_index, repeat_set

if __name__ == '__main__':
    corpus_head, corpus_tail, corpus_title, documents, dataset, Stopword = PrepareData(max_n_samples=1000,Maxcut=50)
    repeat_index, repeat_set = compute_CosSim(corpus_head, corpus_tail, corpus_title, threhold=0.5)
    # repeat_index, repeat_set = compute_simhash(corpus, threhold=15)
    save_result(repeat_index, repeat_set)
    print_quchong(documents, repeat_index)
    print("重复率：",len(repeat_set)/len(documents))


