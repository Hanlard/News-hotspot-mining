# News-hotspot-mining
军事新闻热点挖掘
## 文件说明
1. data/数据集.xlsx: 原始数据集的DEMO，需要替换
2. data/stopwords.txt 中文停用词
3. data/wholewords.txt 金融术语（jieba分词使用）
4. data/repeat_index.pkl 文档去重结果：字典 {"原文档ID":[重复文档ID列表]}
5. data/repeat_set.pkl 文档去重结果：重复文档ID集合
6. RemoveDuplicates.py 去重功能，结果保存为 data/repeat_index.pkl和data/repeat_set.pkl
7. hot_point.py 根据去重文档生成热点，结果保存为 data/热点话题.xlsx
8. simhash.py 提取hash特征（最终没有使用）

## 运行方法
1. 执行 RemoveDuplicates.py 去重，结果保存为 data/repeat_index.pkl和data/repeat_set.pkl
   稀疏矩阵计算使用循环和列表写的，比较慢，建议直接使用去重结果。
2. 执行hot_point.py 进行热点聚类，结果保存为 data/热点话题.xlsx
