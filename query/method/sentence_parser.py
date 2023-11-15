import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

#定义了一个名为LtpParser的类，在初始化方法__init__(self)中加载了
#LTP模型的各个组件（分词器、词性标注器、句法分析器、命名实体识别器和语义角色标注器）。
class LtpParser:
    def __init__(self):
        LTP_DIR = "D:/Environment/Anaconda3/envs/buildkg/Lib/site-packages/pyltp-0.4.0.dist-info/ltp_data_v3.4.0"
        self.segmentor = Segmentor(os.path.join(LTP_DIR, "cws.model"))
        #self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))

        self.postagger = Postagger(os.path.join(LTP_DIR, "pos.model"))
        #self.postagger.load(os.path.join(LTP_DIR, "pos.model"))

        self.parser = Parser(os.path.join(LTP_DIR, "parser.model"))
        #self.parser.load(os.path.join(LTP_DIR, "parser.model"))

        self.recognizer = NamedEntityRecognizer(os.path.join(LTP_DIR, "ner.model"))
        #self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))

        self.labeller = SementicRoleLabeller(os.path.join(LTP_DIR, 'pisrl_win.model'))
        #self.labeller.load(os.path.join(LTP_DIR, 'pisrl_win.model'))

    #format_labelrole函数接收分词结果和词性标注结果作为输入，并使用句法分析器解析句法依存关系。
    # 然后，它使用语义角色标注器获取句子的语义角色信息，并将其整理成字典格式返回。
    def format_labelrole(self, words, postags):
        arcs = self.parser.parse(words, postags)
        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        for role in roles:
            #print(role)
            #roles_dict[role.index] = {arg.name:[arg.name,arg.range.start, arg.range.end] for arg in role.arguments}
            roles_dict[role[0]] = {arg[0]: [arg[0], arg[1][0], arg[1][1]] for arg in role[1]}
        return roles_dict

    #build_parse_child_dict函数接收分词结果、词性标注结果和句法分析结果作为输入，
    #根据句法分析结果构建句法依存关系的子节点字典，并将句法分析结果格式化为指定的列表格式。
    def build_parse_child_dict(self, words, postags, arcs):
        #print(arcs)
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index][0] == index+1:
                    if arcs[arc_index][1] in child_dict:
                        child_dict[arcs[arc_index][1]].append(arc_index)
                    else:
                        child_dict[arcs[arc_index][1]] = []
                        child_dict[arcs[arc_index][1]].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc[0] for arc in arcs]
        relation = [arc[1] for arc in arcs]
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id] 
        for i in range(len(words)):
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i]-1, postags[rely_id[i]-1]]
            format_parse_list.append(a)

        return child_dict_list, format_parse_list

    '''parser_main函数是主要的解析函数，它接收一个句子作为输入。
    首先，它使用分词器将句子分词，然后使用词性标注器获取词性标注结果，接着使用句法分析器解析句法依存关系。
    接下来，它调用build_parse_child_dict函数构建句法依存关系的子节点字典，并调用format_labelrole函数获取语义角色信息。
    最后，它将分词结果、词性标注结果、子节点字典、语义角色信息和句法分析结果的列表格式作为输出返回。'''
    def parser_main(self, sentence):
        words = list(self.segmentor.segment(sentence))
        postags = list(self.postagger.postag(words))
        arcs = self.parser.parse(words, postags)
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)
        roles_dict = self.format_labelrole(words, postags)
        return words, postags, child_dict_list, roles_dict, format_parse_list


if __name__ == '__main__':
    parse = LtpParser()

    sentence = 'toString() 方法用于返回以一个字符串表示的 Number 对象值'
    words, postags, child_dict_list, roles_dict, format_parse_list = parse.parser_main(sentence)
    print(words, len(words))
    print(postags, len(postags))
    print(child_dict_list, len(child_dict_list))
    print(roles_dict)
    print(format_parse_list, len(format_parse_list))