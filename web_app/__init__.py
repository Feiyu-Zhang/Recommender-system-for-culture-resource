# _*_ coding:utf-8 _*_

from flask import Flask, render_template, make_response
from flask.ext.restful import reqparse, abort, Api, Resource
import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# 存储读取语料 一行语料为一个文档
corpus = []
for line in open(r'..\..\data\resourse_jieba_clean.txt', 'r', encoding='utf-8').readlines():
    corpus.append(line.strip())
corpus = corpus[0:100]

# 访问记录id映射
with open(r"..\..\data\visit_res_id372.txt", "r", encoding='utf-8') as recall:
    file_id = recall.read()
ids = file_id.split('\n')

with open(r"..\..\data\res_8_clean.txt", "r", encoding='utf-8') as res_8_clean:
    Resource_id_name = res_8_clean.readlines()

resource_id_name = {resource_id_: resource_name_ for resource_id_, resource_name_ in ((id_name.strip()).split('\t')
                                                                                   for id_name in Resource_id_name)}
# res_num & usr_num
usr_id_lst = list()
res_id_lst = list()
for id in ids:
    lst = id.split('\t')
    if lst[0]:
        usr_id_lst.append(int(lst[0]))
        res_id_lst.append(int(lst[1]))
usr_id_set = set(usr_id_lst)  # 用户id集合
res_id_set = set(res_id_lst)  # 资源id集合
usr_id_lst = list(usr_id_set)  # 用户id
res_id_lst = list(res_id_set)  # 资源id
usr_num = max(usr_id_lst) + 1  # 用户最大id
res_num = max(res_id_lst) + 1  # 资源最大id

# split train & test
train, test = train_test_split(ids, test_size=0.5, random_state=0)
Dic_train = dict()
Dic_test = dict()
for train_i, train_j in enumerate(train):
    train_id = train_j.split('\t')  # lst_id:usr_id-res_id
    int_train_id = list(map(int, train_id))  # int_list:usr_id-res_id
    if int_train_id[0] in Dic_train:
        Dic_train[int_train_id[0]].append(int_train_id[1])
    else:
        Dic_train[int_train_id[0]] = list()
        Dic_train[int_train_id[0]].append(int_train_id[1])
for test_i, test_j in enumerate(test):
    test_id = test_j.split('\t')  # lst_id:usr_id-res_id
    int_test_id = list(map(int, test_id))  # int_list:usr_id-res_id
    if int_test_id[0] in Dic_test:
        Dic_test[int_test_id[0]].append(int_test_id[1])
    else:
        Dic_test[int_test_id[0]] = list()
        Dic_test[int_test_id[0]].append(int_test_id[1])

# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(analyzer='word', stop_words=['系列', '活动', '特别', '技术', '研究', '中国', '世界'])  # max_df=0.95, min_df=3, max_features=30
X = vectorizer.fit_transform(corpus)  # fit_transform是将文本转为词频矩阵
vocabulary = vectorizer.vocabulary_
word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
words = np.asarray(word)  # tag所有词
weight = X.toarray()
size = weight.shape  # 词频矩阵大小
word_num = size[1]  # tag总数

# LDA算法
model = lda.LDA(n_topics=10, n_iter=100, random_state=0)
model.fit(np.asarray(weight))
topic_word = model.topic_word_  # 主题-词 （Topic-Word）分布
doc_topic = model.doc_topic_  # 文档-主题（Document-Topic）分布
doc_word_weight = np.zeros((res_num, word_num))  # doc_word为资源-标签权重
res_profile = dict()

for m, n in enumerate(res_id_lst):  # m:LDA中doc_topic的doc下标 n:resourseID
    p = 1  # p:一篇文章的主题数目5
    n = int(n)
    reversed_doc = np.argsort(doc_topic[m])[:-(p + 1):-1]  # p个主题数目下标
    for i, j in enumerate(reversed_doc):
        topic_most_pr = j  # 第i个主题下标j
        q = 19  # q:一个主题的词数目30
        topic_dist = topic_word[topic_most_pr]  # 主题下词的相关性
        reversed_topic = np.argsort(topic_dist)[::-1]  # 相关性由大到小的词下标排序
        for u in range(q):
            doc_word_weight[n][reversed_topic[u]] = doc_word_weight[n][reversed_topic[u]] + \
                                                    doc_topic[m][topic_most_pr] * \
                                                    topic_word[topic_most_pr][reversed_topic[u]]
        res_profile[n] = words[reversed_topic[0:5]]

# 利用访问记录，生成用户画像
usr_tag = np.zeros((usr_num, word_num))  # 4379usr_num 5902word_num
for id_visit in train:
    lst_visit = id_visit.split('\t')
    if lst_visit[0]:
        usr_id = int(lst_visit[0])
        res_id = int(lst_visit[1])
        usr_tag[usr_id][:] += doc_word_weight[res_id]
usr_profile = dict()
for usr_profile_id in usr_id_lst:
    reversed_tag = np.argsort(usr_tag[usr_profile_id])[:-11:-1]
    usr_profile[usr_profile_id] = words[reversed_tag]

# 计算用户与资源之间的相似度
K = cosine_similarity(usr_tag[usr_id_lst], doc_word_weight[res_id_lst], dense_output=True)

# 计算推荐准确率
num_rec = 10  # 推荐数目
precisions = list()
recalls = list()
K_ss = list()
rec_lists = dict()
for i_K, usr_id_rec in enumerate(usr_id_lst):
    rec_lists[i_K] = list()
    correct = 0
    succeed_rec = 0
    rec_ids = np.argsort(K[i_K])[:-(num_rec + 1):-1]
    res_recommend = list()  # res_recommend:资源id
    for rec_id in rec_ids:
        res_recommend.append(res_id_lst[rec_id])
    for id_rec in res_recommend:
        if id_rec in Dic_train[usr_id_rec]:
            continue
        elif id_rec in Dic_test[usr_id_rec]:
            correct += 1
            succeed_rec += 1
            rec_lists[i_K].append(id_rec)
        else:
            succeed_rec += 1
            rec_lists[i_K].append(id_rec)
    if succeed_rec:
        precisions.append(correct/succeed_rec)
        recalls.append(correct/len(Dic_test[usr_id_rec]))
        K_s = cosine_similarity(doc_word_weight[rec_lists[i_K]], dense_output=True)
        K_row = len(K_s)
        K_ss.append((np.sum(K_s) - K_row) / (K_row * (K_row - 1)))
precision = np.mean(precisions)
recall = np.mean(recalls)
F1 = 2*precision*recall/(precision+recall)
ILS = np.mean(K_ss)
print('precision:{:.4f}\nrecall:{:.4f}\nF1:{:.4f}\nILS:{:.4f}'.format(precision, recall, F1, ILS))

app = Flask(__name__)
api = Api(app)


@app.template_global(name='zip')
def _zip(*args, **kwargs):  # to not overwrite builtin zip in globals
    return __builtins__.zip(*args, **kwargs)


def abort_if_todo_doesnt_exist(userid):
    if userid not in rec_lists:
        abort(404, message="User {} doesn't exist".format(userid))


parser = reqparse.RequestParser()
parser.add_argument('userid')
parser.add_argument('resourceid', action='append')


class home(Resource):
    def get(self):
        return make_response(render_template('index.html'))


class result(Resource):
    def get(self):
        return make_response(render_template('result.html'))

    def post(self):
        args = parser.parse_args()
        user_id = int(args['userid'])-1
        recommend_id = rec_lists[user_id]
        profile = usr_profile[usr_id_lst[user_id]]
        resourse_profile = list()
        for r_id in recommend_id:
            resourse_profile.append(res_profile[r_id])
        resource_name = [resource_id_name[str(resource_rec_id)] for resource_rec_id in recommend_id]
        return make_response(render_template('result.html', rec_id=recommend_id, usr_id=user_id+1, profile=profile,
                                             resource_name=resource_name))

                                             
class load(Resource):
    def get(self):
        return make_response(render_template('load.html'))

    def post(self):
        args = parser.parse_args()
        new_profile = np.zeros((1, word_num))
        for new_user_resource_id in args['resourceid']:
            new_profile += doc_word_weight[int(new_user_resource_id)]
        reversed_tag = np.argsort(new_profile[0])[:-11:-1]
        new_user_profile = words[reversed_tag]
        cos_similarity = cosine_similarity(new_profile, doc_word_weight[res_id_lst], dense_output=True)
        new_user_rec_list = np.argsort(cos_similarity[0])[:-6:-1]  # 推荐列表资源
        new_user_rec_list = new_user_rec_list.tolist()
        user_id = int(args['userid'])
        resourse_profile = list()  # 考虑业务要求已开发，但暂未使用
        for r_id in new_user_rec_list:
            resourse_profile.append(res_profile[res_id_lst[r_id]])
        reversed_tag = np.argsort(usr_tag[user_id])[:-11:-1]
        profile = words[reversed_tag]
        resource_name = [resource_id_name[str(res_id_lst[resource_rec_id])] for resource_rec_id in new_user_rec_list]
        return make_response(render_template('result.html', rec_id=new_user_rec_list, usr_id=user_id,
                                             profile=new_user_profile, resource_name=resource_name))


api.add_resource(home, '/index')
api.add_resource(result, '/result')
api.add_resource(load, '/load')

if __name__ == '__main__':
    app.run(debug=True)
