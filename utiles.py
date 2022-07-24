from transformers import BertTokenizer

#读取训练集、测试集、验证集，转化数据形式[(sent,tag)]
def read_file(f_path):
    with open(f_path,'r',encoding='utf8') as f:
        lines = [line.split('\n')[0] for line in f.readlines() if len(line.strip())!=0]
        sent_label = []
        words = [line.split('\t')[0] for line in lines]
        tags = [line.split('\t')[1] for line in lines]
        t_word = []
        t_tag = []
        for word,tag in zip(words,tags):
            if word != '。':
                t_word.append(word)
                t_tag.append(tag)
            else:
                sent_label.append((t_word,t_tag))
                t_word = []
                t_tag = []
    return sent_label

def convert_format(examples,max_len,label_map,model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    token_ids = []
    label_ids = []
    for example in examples:
        sent,label = example
        text_len = len(sent)
        if text_len > max_len-2:  #截断,[CLS],[SEP]
            sent = sent[:max_len-2]
            label = label[:max_len-2]
        sent.insert(0,'[CLS]')
        label.insert(0,'[CLS]')
        sent.append('[SEP]')
        label.append('[SEP]')
        token_id = tokenizer.convert_tokens_to_ids(sent)
        label_id = [label_map[tag] for tag in label]
        if len(token_id) < max_len:
            token_id = token_id + [0]*(max_len-len(token_id))
            label_id = label_id + [0]*(max_len-len(label_id))
        token_ids.append(token_id)
        label_ids.append(label_id)
    return token_ids,label_ids
    


        








