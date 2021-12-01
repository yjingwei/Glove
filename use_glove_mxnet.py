from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text

#关于fasttext的词向量的使用
# text_data="hello world \n hello nice world \n hi world"
# counter=text.utils.count_tokens_from_str(text_data)
#
# print(counter)
# #根据数据集建立词典
# my_vocab=text.vocab.Vocabulary(counter)
# #为该词典的词载入fasttext词向量，使用Simple English的预训练词向量
# my_embedding=text.embedding.create('fasttext',pretrained_file_name='wiki.simple.vec',vocabulary=my_vocab)
#
# print(len(my_embedding))#5
# #如果词没在词典中，则词向量默认为0
# a=my_embedding.get_vecs_by_tokens('beautiful')[:10]
# print(a)
#
# a=my_embedding.get_vecs_by_tokens(['hello','world']).shape#(2, 300)
# print(a)
#
# a=my_embedding.to_indices(['hello','world'])#[2, 1]
# print(a)

###下面是glove词向量的使用
a=text.embedding.get_pretrained_file_names('glove')
print(a)

glove_6B_50d=text.embedding.create('glove',pretrained_file_name='glove.6B.50d.txt')
print(len(glove_6B_50d))#400001 那个多出来的1就是unknow

print(glove_6B_50d.token_to_idx['beautiful'])
print(glove_6B_50d.idx_to_token[3367])
print(glove_6B_50d.vec_len)


def cos_sim(x,y):
    return nd.dot(x,y)/(nd.norm(x)*nd.norm(y))

def norm_vecs_by_row(x):
    return x/nd.sqrt(nd.sum(x*x,axis=1).reshape((-1,1)))

def get_knn(token_embeding,k,word):
    word_vec=token_embeding.get_vecs_by_tokens([word]).reshape((-1,1))#[50，1]
    vocab_vecs=norm_vecs_by_row(token_embeding.idx_to_vec)#[400001,50]
    dot_prod=nd.dot(vocab_vecs,word_vec)#[400001，1]
    indices=nd.topk(dot_prod.reshape((len(token_embeding),)),k=k+2,ret_typ='indices')
    indices=[int(i.asscalar()) for i in indices]#X.asscalar()将向量X转换成标量


    return token_embeding.to_tokens(indices[2:])

print(get_knn(glove_6B_50d,5,'happy'))
