代码主要完成辱骂文本对抗。 其中main.py主要是训练过程。run.py为对抗过程。

其中文本分类采用bilstm+attention+softmax模型。 对抗过程是采用黑盒攻击法。
产生对抗样本的过程： 采用left+right的预测方法。

举例：他真不要脸啊。 
left数据预处理：

1.他 
2.他 真
3.他 真 不要脸 
4.他 真 不要脸 啊 

right数据预处理成： 
1.他 真 不要脸 啊 
2.真 不要脸 啊 
3.不要脸 啊 
4.啊 

将上述left生成的4个句子以及right生成的4个句子均进行辱骂性质预测。 left中当出现不要脸时，句子“他 真 不要脸”变成辱骂性质，故识别出“不要脸”词是导致句子为辱骂性质的辱骂词，进行替换即可。 同理right中前三个句子均被标识为辱骂性质，只有最后句子“啊”被识别为非辱骂性质，故识别出“不要脸”词是导致句子为辱骂性质的辱骂词。进行替换即可。

代码中同时考虑了一句话中同时出现多个辱骂词的情况，每次遍历一遍left句组识别出最左面的辱骂词，遍历一遍right识别出最右面的辱骂词。故代码中加入循环，直到句子彻底变为无辱骂性质的句子结束。 

该代码思路在阿里举办的辱骂文本对抗2020比赛中取得了第19名成绩。 
成绩不是很好，主要原因总结如下： 1.我的训练数据辱骂词较为单薄，不能很好的覆盖比赛模型训练数据辱骂词情况。 2.模型简单，需要改进的东西很多。 由于辱骂数据较为敏感，故数据没有上传
