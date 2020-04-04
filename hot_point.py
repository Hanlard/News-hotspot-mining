from RemoveDuplicates import *
import os
import pandas as pd
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from textrank4zh import TextRank4Sentence
from gensim.models import word2vec
import jieba
jieba.load_userdict("data/wholewords.txt")
from jieba.analyse import extract_tags, textrank
from word2vec_textrank import summarize
import joblib

def Keywords(sen, mode="extract_tags",not_keywords=[],N=5):
    M = len(not_keywords)
    if mode == "extract_tags":
        keywords_ = extract_tags(clean(sen), topK=M+N,allowPOS=('n','v','vi','nr', 'ns', 'nt', 'ni', 't', 'x'))
    keywords = [k for k in keywords_ if k not in not_keywords][:N]
    return  keywords


def feature_extraction(series, vectorizer='CountVectorizer', vec_args=None):
    """
    对原文本进行特征提取
    :param series: pd.Series，原文本
    :param vectorizer: string，矢量化器，如'CountVectorizer'或者'TfidfVectorizer'
    :param vec_args: dict，矢量化器参数
    :return: 稀疏矩阵
    """
    vec_args = {'max_df': 1.0, 'min_df': 1, 'ngram_range': (1,2)} if vec_args is None else vec_args
    vec_args_list = ['%s=%s' % (i[0],
                                "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                ) for i in vec_args.items()]
    vec_args_str = ','.join(vec_args_list)
    vectorizer1 = eval("%s(%s)" % (vectorizer, vec_args_str))
    matrix = vectorizer1.fit_transform(series)
    return matrix


def get_cluster(matrix, cluster='DBSCAN', cluster_args=None):
    """
    对数据进行聚类，获取训练好的聚类器
    :param matrix: 稀疏矩阵
    :param cluster: string，聚类器
    :param cluster_args: dict，聚类器参数
    :return: 训练好的聚类器
    """

    cluster_args = {'eps': 0.25, 'min_samples': 5, 'metric': 'cosine'} if cluster_args is None else cluster_args
    cluster_args_list = ['%s=%s' % (i[0],
                                    "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                    ) for i in cluster_args.items()]
    cluster_args_str = ','.join(cluster_args_list)
    cluster1 = eval("%s(%s)" % (cluster, cluster_args_str))
    cluster1 = cluster1.fit(matrix)
    return cluster1

def get_labels(cluster):
    """
    获取聚类标签
    :param cluster: 训练好的聚类器
    :return: list，聚类标签
    """
    labels = cluster.labels_
    return labels

def TfIdf_keywords(corpus):
    vectorizer = CountVectorizer()
    count = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count)
    sort = np.argsort(tfidf.toarray(), axis=1)[:, -5:]  # 将二维数组中每一行按升序排序，并提取每一行的最后五个(即数值最大的的五个)
    names = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    keyWords = pd.Index(names)[sort]
    return keyWords

def get_key_sentences(text, num=1):
    """
    利用textrank算法，获取文本摘要
    :param text: string，原文本
    :param num: int，指定摘要条数
    :return: string，文本摘要
    """
    tr4s = TextRank4Sentence(stop_words_file="data/stopwords.txt",
                             delimiters=['?', '!', ';', '？', '！', '。', '；',
                                         '……', '…', '\n',':','：',
                                         ]
                             )
    tr4s.analyze(text=text, lower=True, source='all_filters',pagerank_config = {'alpha': 0.85,})
    abstract = '\n'.join([item.sentence for item in tr4s.get_key_sentences(num=num)])
    return abstract

def Merge_process(JuLei,dataset):
    Hotpoint = {"话题ID": [],"公司名称":[], "话题概述":[], "话题热词": [], "新闻标题": [], "新闻正文": [], "新闻热词": [], "链接":[]}
    with open("data/not_keywords.txt", "r", encoding="utf-8") as f:
        not_keywords = set(f.read().splitlines())
    for class_index in tqdm(JuLei):
        ##	目标公司	简称	标题	正文	来源	链接	时间
        text = ""
        titles = ""
        keywords_ = []
        for i, index in enumerate(JuLei[class_index]):
            Hotpoint["公司名称"].append(dataset['目标公司'].iloc[index])
            Hotpoint["链接"].append(dataset['链接'].iloc[index])
            Hotpoint["话题ID"].append(class_index)
            data = dataset['正文'].iloc[index]  # 正文
            text += data+"\n"
            Hotpoint["新闻正文"].append(data)
            ## 提取关键词
            keywords = Keywords(data, mode="extract_tags",not_keywords=not_keywords,N=5)
            keywords_ = keywords_ + keywords
            Hotpoint["新闻热词"].append(" ".join(keywords))
            title = dataset['标题'].iloc[index]
            # title = clean(title)
            titles += title+"\n"
            Hotpoint["新闻标题"].append(title)

        ## 聚类摘要-使用摘要的摘要
        ## 新闻标题摘要
        topic_abstract = get_key_sentences(titles, num=3)

        topic_keywords = extract_tags(",".join(keywords_), topK=10)
        for _ in JuLei[class_index]:
            Hotpoint["话题热词"].append(" ".join(list(topic_keywords)))
            Hotpoint["话题概述"].append(topic_abstract)
    Hotpoint = pd.DataFrame(Hotpoint)
    return Hotpoint

def Merge(Hotpoint,JuLei,dataset,firstK=2):
    print("合并重复类...")
    ## 按照聚类热词合并相似聚类
    tipic_ID = Hotpoint["话题ID"]
    topic_keywords = Hotpoint["话题热词"]
    t_k = {}
    for ID in JuLei:
        index = tipic_ID[tipic_ID.values==ID].index[0]
        t_k[ID] = topic_keywords.iloc[index].split()
    num_IDs = len(t_k)
    Combine = {}
    Combined = set()
    for i in range(num_IDs-1):
        keywords_ID_i = set(t_k[i][:firstK])
        for j in range(i+1,num_IDs):
            if j not in Combined:# j没有和任意ID合并
                keywords_ID_j = set(t_k[j][:firstK])
                if keywords_ID_i & keywords_ID_j: #有交集，可以合并
                    if i not in Combine:
                        Combine[i]=set()
                    Combine[i].add(j)
                    Combined.add(j)
    New_JuLei_ = {}
    for ID in JuLei:
        if ID in Combined:
            continue
        elif ID in Combine:
            cu_ID = JuLei[ID]
            for id in Combine[ID]:
                cu_ID = cu_ID + JuLei[id]
            New_JuLei_[ID] = cu_ID
        else:
            New_JuLei_[ID] = JuLei[ID]
    New_JuLei = {i:New_JuLei_[key] for i ,key in enumerate(New_JuLei_)}
    Hotpoint = Merge_process(JuLei=New_JuLei,dataset=dataset)

    return Hotpoint
def Generate_Hotpoint(JuLei, dataset, Stopwords):
    print("生成热点与摘要...")
    Hotpoint = {"话题ID": [],"公司名称":[], "话题概述":[], "话题热词": [], "新闻标题": [], "新闻正文": [], "新闻热词": [], "链接":[]}
    with open("data/not_keywords.txt", "r", encoding="utf-8") as f:
        not_keywords = set(f.read().splitlines())
    for class_index in tqdm(JuLei):
        ##	目标公司	简称	标题	正文	来源	链接	时间
        # corpus_keywords = {}
        text = ""
        titles = ""
        abstracts = ""
        keywords_ = []
        for i, index in enumerate(JuLei[class_index]):
            Hotpoint["公司名称"].append(dataset['目标公司'].iloc[index])
            Hotpoint["链接"].append(dataset['链接'].iloc[index])
            Hotpoint["话题ID"].append(class_index)
            data = dataset['正文'].iloc[index]  # 正文
            text += data+"\n"
            Hotpoint["新闻正文"].append(data)
            ## 提取关键词
            # keywords = textrank(" ".join(del_stopword(data, Stopwords)), topK=5,allowPOS=("n", "v", 'vi', 'nr', 'ns', 'nt', 'ni', 't', 'x'))
            # keywords = textrank(data, topK=5,
            #                     allowPOS=("n", "v", 'vi', 'nr', 'ns', 'nt', 'ni', 't', 'x'))
            keywords = Keywords(data, mode="extract_tags",not_keywords=not_keywords,N=5)
            keywords_ = keywords_ + keywords
            Hotpoint["新闻热词"].append(" ".join(keywords))
            title = dataset['标题'].iloc[index]
            # title = clean(title)
            titles += title+"\n"
            Hotpoint["新闻标题"].append(title)
            # 内容摘要 标题+前80字
            # abstracts += get_key_sentences(data[:min(80,len(data))],num=3)+"\n"
            # abstracts += "\n".join(summarize(data[:min(500,len(data))],n=3))+"\n"

        ## 聚类摘要-使用摘要的摘要
        ## 新闻标题摘要
        topic_abstract = get_key_sentences(titles, num=3)
        ## 新闻摘要的摘要
        # topic_abstract = get_key_sentences(abstracts,num=1)
        # topic_abstract = "\n".join(summarize(abstracts,n=3))

        ## 聚类主题词-使用全文查找主题词 extract_tags/textrank
        # topic_keywords = extract_tags("".join(del_stopword(text, Stopwords)), topK=5, allowPOS=("n", "v", 'vi', 'nr', 'ns', 'nt', 'ni', 't', 'x'))
        # topic_keywords = extract_tags(text, topK=10,
        #                               allowPOS=("n", "v", 'vi', 'nr', 'ns', 'nt', 'ni', 't', 'x'))
        topic_keywords = extract_tags(",".join(keywords_), topK=10)
        for _ in JuLei[class_index]:
            Hotpoint["话题热词"].append(" ".join(list(topic_keywords)))
            Hotpoint["话题概述"].append(topic_abstract)
    Hotpoint = pd.DataFrame(Hotpoint)
    return Hotpoint

def save_hotpoint(JuLei, dataset, Stopwords):
    Hotpoint=Generate_Hotpoint(JuLei, dataset,Stopwords)
    Hotpoint = Merge(Hotpoint,JuLei,dataset,firstK=2)
    Hotpoint.to_excel("data/热点话题.xlsx",index=False)
    print("生成热点话题："+"/data/热点话题.xlsx")

def PrintHotpoint(JuLei,dataset, Stopwords):
    for class_index in JuLei:
        print("——————————————————【聚类】—class_index：",class_index,"——————————————————")
        corpus = []
        corpus_keywords={}
        for i,index in enumerate(JuLei[class_index]):
            data = dataset['正文'].iloc[index]
            keywords = textrank(" ".join(del_stopword(data, Stopwords)),topK=5,allowPOS=("n","v",'vi','nr','ns','nt','ni','t'))
            # keywords = extract_tags(" ".join(del_stopword(data, Stopwords)), topK=5, allowPOS=("n", "v", 'vi', 'nr', 'ns', 'nt', 'ni', 't'))
            print("【关键词】", keywords, "ID -", index, "【正文】", data)
            for word in keywords:
                if word not in corpus_keywords:
                    corpus_keywords[word]=0
                corpus_keywords[word] += 1
        corpus_keywords = pd.Series(corpus_keywords)
        corpus_keywords=corpus_keywords.sort_values(ascending=False)
        print("簇主题词")
        print(corpus_keywords[:5])

if __name__ == '__main__':
    if os.path.exists('data/dataset_qx.pkl'):
        print("Reading dataset ...")
        dataset = joblib.load('data/dataset_qx.pkl')
        with open('data/stopwords.txt', "r", encoding="utf-8") as f:
            Stopwords = set(f.read().splitlines())
    else:
        dataset, Stopwords = PrepareData(max_n_samples=None,onlyload=True)
        dataset = CleanData(dataset)
        print("数据清洗完成")
        repeat_index, repeat_set = load_result()
        # print_quchong(list(dataset['正文'].values), repeat_index)
        dataset = dataset.drop(index=list(repeat_set))
        print(dataset.shape)
        print("去重完成")
        joblib.dump(dataset, 'data/dataset_qx.pkl')
    ## 按正文/标题聚类
    matrix = feature_extraction(dataset['正文'], vectorizer='TfidfVectorizer', vec_args={'max_df': 0.95,
                                                                                       'min_df': 0.01,
                                                                                       "stop_words": Stopwords,
                                                                                       'ngram_range': (1, 2)
                                                                                       })
    print("提取特征完成")
    ## 距离选项 'cityblock', 'cosine' and 'euclidean'
    ## eps：分类的精准度（base：0.25） min_smples: 聚类的密度（5）
    cluster1 = get_cluster(matrix, cluster='DBSCAN', cluster_args = {'eps': 0.25, 'min_samples': 5, 'metric': 'cosine','n_jobs':-1} )
    print("初步聚类完成")
    content_labels = list(get_labels(cluster1))
    JuLei={}
    for index,class_index in enumerate(content_labels):
        if class_index > -1:
            if class_index not in JuLei:
                JuLei[class_index]=[]
            JuLei[class_index].append(index)

    ## 保存
    save_hotpoint(JuLei, dataset, Stopwords)
    # PrintHotpoint(JuLei, dataset, Stopwords)