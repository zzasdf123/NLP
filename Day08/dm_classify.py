# -*- coding: utf-8 -*-
import fasttext

# todo:1.数据未清洗前
# 1.初始模型训练
# model = fasttext.train_supervised('./fasttext_data/cooking.train')
# 2.模型保存
# model.save_model('./cooking.model')
# 3.加载模型
# model = fasttext.load_model('./cooking.model')
# 4.模型预测
# result = model.predict('How do I make a hamburger?')
# print(result)
# 5.查看模型在验证集上的表现
# result = model.test('./fasttext_data/cooking.valid')
# 样本数量+精确率+召回率
# print(f'数据未经过处理的验证集的结果：{result}')

# todo：2.数据清洗之后
# # 1.初始模型训练
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train')
# # 2.模型保存
# model.save_model('./cooking_new.model')
# # 3.加载模型
# model = fasttext.load_model('./cooking_new.model')
# # 4.模型预测
# result = model.predict('How do I make a hamburger ?')
# print(result)
# # 5.查看模型在验证集上的表现
# result = model.test('./fasttext_data/cooking_pre.valid')
# # 样本数量+精确率+召回率
# print(f'数据经过预处理的验证集的结果：{result}')

# todo:3.增加训练轮数
# # 1.训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train',epoch=25)
# # 2.查看模型在训练集上的表现
# result = model.test('./fasttext_data/cooking_pre.valid')
# print(f'增加训练轮数的训练集的结果：{result}')

# todo:4.修改学习率
# # 1.训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train',epoch=25,lr=1.0)
# # 2.查看模型在训练集上的表现
# result = model.test('./fasttext_data/cooking_pre.valid')
# print(f'修改学习率的训练集的结果：{result}')

#todo:5.增加n-gram特征
#1.训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train',epoch=25,lr=1.0,wordNgrams=2)
# 2.查看模型在训练集上的表现
# result = model.test('./fasttext_data/cooking_pre.valid')
# print(f'增加n-gram特征的训练集的结果：{result}')

# todo:6.修改损失方式loss
# # 1.训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train',epoch=25,lr=1.0,wordNgrams=2,loss='hs')
# # 2.查看模型在训练集上的表现
# result = model.test('./fasttext_data/cooking_pre.valid')
# print(f'修改损失方式loss的训练集的结果：{result}')

# todo:7.自动超参数调优
# # 1.训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train',autotuneValidationFile='./fasttext_data/cooking_pre.valid',autotuneDuration=600)
# # 2.查看模型在训练集上的表现
# result = model.test('./fasttext_data/cooking_pre.valid')
# print(f'自动超参数调优的训练集的结果：{result}')

# todo:8.修改损失函数为ova
# 1.训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train',epoch=25,lr=0.2,wordNgrams=2,loss='ova')
# 2.保存模型
# model.save_model('./cooking_new1.model')
# 3.加载模型
model = fasttext.load_model('./cooking_new1.model')
# 4.模型预测
result1 = model.predict('Which baking dish is best to bake a banana bread ?',k=-1,threshold=0.5)
print(result1)
# 5.查看模型在训练集上的表现
# result = model.test('./fasttext_data/cooking_pre.valid')
# print(f'修改损失函数为ova的训练集的结果：{result}')