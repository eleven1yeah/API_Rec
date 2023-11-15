#从当前文件夹导入
from sentence_parser import *

import re

class TripleExtractor:
    def __init__(self):
        self.parser = LtpParser()

    #将文本内容切分为句子，输入参数content，使用正则表达式和列表推导式将输入的文本切分成句子，返回sentence
    def split_sents(self, content):
        return [sentence for sentence in re.split(r'[？?！!。；;：:\n\r]', content) if sentence]

    #定义规则函数ruler1，根据角色信息和词性标注来提取主谓宾结构的三元组
    def ruler1(self, words, postags, roles_dict, role_index):
        #获取给定角色索引role_index处的词语，作为谓词（动词）
        v = words[role_index]
        #获取给定角色索引role_index在角色字典中的信息
        role_info = roles_dict[role_index]
        if 'A0' in role_info.keys() and 'A1' in role_info.keys():
            s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2]+1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2]+1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            if s  and o:
                return '1', [s, v, o]
        return '4', []

    #triple extraction
    #定义规则函数ruler2，根据依存句法分析结果和角色信息来提取主谓宾结构的三元组
    def ruler2(self, words, postags, child_dict_list, arcs, roles_dict):
        svos = []
        for index in range(len(postags)):
            tmp = 1

            #如果当前词语的索引在角色字典中存在，则调用ruler1来提取主谓宾三元组，并将返回的结果加入到svos中。
            if index in roles_dict:
                flag, triple = self.ruler1(words, postags, roles_dict, index)
                if flag == '1':
                    svos.append(triple)
                    tmp = 0
            if tmp == 1:

                if postags[index]:
                    #获取当前词语的子节点字典
                    child_dict = child_dict_list[index]

                    if 'SBV' in child_dict and 'VOB' in child_dict:
                        r = words[index]
                        e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                        svos.append([e1, r, e2])

                    #获取当前词语与其父节点之间的依存关系和父节点的索引
                    relation = arcs[index][0]
                    head = arcs[index][2]
                    if relation == 'ATT':
                        if 'VOB' in child_dict:
                            e1 = self.complete_e(words, postags, child_dict_list, head - 1)
                            r = words[index]
                            e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                            temp_string = r + e2
                            if temp_string == e1[:len(temp_string)]:
                                e1 = e1[len(temp_string):]
                            if temp_string not in e1:
                                svos.append([e1, r, e2])

                    if 'SBV' in child_dict and 'CMP' in child_dict:
                        e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        cmp_index = child_dict['CMP'][0]
                        r = words[index] + words[cmp_index]
                        if 'POB' in child_dict_list[cmp_index]:
                            e2 = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                            svos.append([e1, r, e2])
        return svos

    #定义了辅助方法complete_e，用于递归地获取一个词语及其修饰词的完整词语。
    def complete_e(self, words, postags, child_dict_list, word_index):
        child_dict = child_dict_list[word_index]
        prefix = ''
        #如果子节点字典中包含'ATT'（修饰关系），则递归地获取修饰词的完整词语，并将其添加到前缀prefix中
        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix += self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
        postfix = ''
        if postags[word_index] == 'v':
            #递归地获取宾语的完整词语
            if 'VOB' in child_dict:
                postfix += self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
            #递归地获取主语的完整词语
            if 'SBV' in child_dict:
                prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

        return prefix + words[word_index] + postfix

    def triples_main(self, content):
        sentences = self.split_sents(content)
        svos = []
        for sentence in sentences:
            words, postags, child_dict_list, roles_dict, arcs = self.parser.parser_main(sentence)
            svo = self.ruler2(words, postags, child_dict_list, arcs, roles_dict)
            svos += svo

        return svos

def test():
    content5 = ''
    #content5 = 'toString() 方法用于返回以一个字符串表示的 Number 对象值'
    extractor = TripleExtractor()
    svos = extractor.triples_main(content5)
    print('svos', svos)

#if __name__ == '__main__':
#    test()

