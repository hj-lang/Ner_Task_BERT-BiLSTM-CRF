import json
import os
from sklearn.model_selection import train_test_split

#读取中文医药文件
def read_chinese_medical_file(file_path):
    label_dict = {'药物':'DRUG',
              '解剖部位':'BODY',
              '疾病和诊断':'DISEASES',
              '影像检查':'EXAMINATIONS',
              '实验室检验':'TEST',
              '手术':'TREATMENT'}
    f = open(file_path,'r',encoding='utf-8-sig')
    sentences = []
    labels = []

    while True:
        line = f.readline()
        if not line:
            break
        line_dict = json.loads(line)
        original_text = list(line_dict['originalText'])
        sentences.append(original_text)
        entities = line_dict['entities']
        label = ["O"]*len(original_text)
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            label_type = entity['label_type']
            b_label = 'B-'+label_dict[label_type]
            i_label = 'I-'+label_dict[label_type]
            label[start_pos] = b_label
            for pos in range(start_pos+1,end_pos):
                label[pos] = i_label
        labels.append(label)
    return sentences,labels

#存储预处理数据
def save_data(file_path,sentences,labels,type):
    file_name = type+'.txt'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    path = os.path.join(file_path,file_name)
    with open(path, 'w') as f:
        for sent, label in zip(sentences, labels):
            sent_len = len(sent)
            for i in range(sent_len):
                f.write(sent[i])
                f.write('\t')
                f.write(label[i])
                f.write('\n')
if __name__=='__main__':
    #中文医药
    train_part1_path = '../ner_task/raw_data/chinese_medical/subtask1_training_part1.txt'
    train_part2_path = '../ner_task/raw_data/chinese_medical/subtask1_training_part2.txt'
    test_path = '../ner_task/raw_data/chinese_medical/subtask1_test_set_with_answer.json'
    output_path = '../ner_task/data/chinese_medical'
    sentences1,labels1 = read_chinese_medical_file(train_part1_path)
    sentences2,labels2 = read_chinese_medical_file(train_part2_path)
    test_sentence,test_label = read_chinese_medical_file(test_path)
    sentences1.extend(sentences2)
    labels1.extend(labels2)
    train_sentence,val_sentence,train_label,val_label = train_test_split(sentences1,labels1,test_size=0.2)
    save_data(output_path,train_sentence,train_label,'train')
    save_data(output_path,val_sentence,val_label,'valid')
    save_data(output_path,test_sentence,test_label,'test')

