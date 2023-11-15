#构建知识图谱 并将知识图谱放入neo4j
import os
import json
from py2neo import Graph,Node

functions=["test1","test2"]
rels_use=[("test1","test2")]
# rels_use 是一个元组的列表，其中有一个元组 ("test1","test2")，表示函数 "test1" 依赖于函数 "test2"

class MedicalGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        #self.data_path = os.path.join(cur_dir, 'data/medical2.json')
        #self.g = Graph("http://localhost:7474", username="neo4j", password="Zhaoy000710@")
        self.g = Graph("http://localhost:7474", auth=("neo4j", "Zhaoy000710@"))

    '''建立节点'''
    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
            count += 1
            print(count, len(nodes))
        return

    '''创建知识图谱实体节点类型'''
    def create_graphnodes(self):
        self.create_node('function',functions)
        self.create_node('class',classes)
        self.create_node('parameter',parameters)
        self.create_node('codefile',codefile)
        return

    '''创建实体关系边'''
    def create_rel_use(self):
        self.create_relationship('function', 'function', rels_use, 'use_function', '调用关系')
        return
    # 'function' 表示源节点的类型为函数。
    # 'function' 表示目标节点的类型也为函数。
    # rels_use 是一个列表，其中存储了多个元组，每个元组表示一个函数之间的依赖关系。这个列表在方法外部定义，是通过上下文传进来的。
    # 'use_function' 表示依赖关系的类型为 use_function，即调用关系。
    # '调用关系' 表示依赖关系的标签为 调用关系。

    def creat_rel_class_variable(self):
        self.create_relationship('class', 'parameter', rel_class_variable, 'have_variable', '包含变量')
        return

    def creat_rel_function_parameter(self):
        self.create_relationship('function', 'parameter', rel_function_parameter, 'parameter', '函数参数')

    def creat_rel_inherit(self):
        self.create_relationship('class','class', rel_inherit,'inherit',"继承")
        return

    def create_rel_use_class(self):
        self.create_relationship('class','codefile',rel_use_class,'call_class','类被引用')

    def create_rel_use_function(self):
        self.create_relationship('function','codefile',rel_use_function,'call_function','函数被引用')
        return

    '''创建实体关联边，创建节点间依赖关系'''
#首先将 edges 中的元组转化成字符串格式，并去重。
# 然后遍历去重后的元组，对于每个元组，解析出源节点和目标节点的名称，
# 然后构造一个 Cypher 查询语句，使用 run 方法执行该查询，并将执行结果打印出来。
# 其中，查询语句中的 %s 表示占位符，用于动态插入变量。
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            # 将每对元素以 ### 分割 并添加到set_edges
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            # 遍历每对元素
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]

            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return


# 读取文件内容
with open("analyse/functions.txt", "r") as f_functions:
    content = f_functions.readlines()
    # 初始化一个名为functions的空列表，将在其中添加函数名
    functions = []
    for line in content:
        functions.append(line.strip())

with open("analyse/class.txt", "r") as f_class:
    content = f_class.readlines()
    classes = []
    for line in content:
        classes.append(line.strip())

with open("analyse/parameter.txt", "r") as f_parameter:
    content = f_parameter.readlines()
    parameters = []
    for line in content:
        parameters.append(line.strip())

with open("analyse/codefile.txt", "r") as f_codefile:
    content = f_codefile.readlines()
    codefile = []
    for line in content:
        codefile.append(line.strip())

with open("analyse/rel_use.txt", "r") as f_rel_use:
    content = f_rel_use.readlines()
    rels_use = []
    for line in content:
        # 对于每一行，使用 eval 函数将其解析成一个字典，并将结果赋值给变量 dic
        dic = eval(line)
        for key in dic:
            rels_use.append((dic[key], key))
    # 输出结果：[('PythonProceduralTest.test', 'open'),....

with open("analyse/rel_class_variable.txt", "r") as f_rel_class_variable:
    content=f_rel_class_variable.readlines()
    rel_class_variable=[]
    for line in content:
        dic=eval(line)
        for key in dic:
            rel_class_variable.append((dic[key],key))

with open("analyse/rel_function_parameter.txt", "r") as f_rel_function_parameter:
    content=f_rel_function_parameter.readlines()
    rel_function_parameter=[]
    for line in content:
        dic=eval(line)
        for key in dic:
            rel_function_parameter.append((dic[key],key))

with open("analyse/rel_function_return.txt", "r") as f_rel_function_return:
    content=f_rel_function_return.readlines()
    rel_function_return=[]
    for line in content:
        dic=eval(line)
        for key in dic:
            rel_function_return.append((key,dic[key]))

with open("analyse/rel_inherit.txt", "r") as f_rel_inherit:
    content=f_rel_inherit.readlines()
    rel_inherit=[]
    for line in content:
        dic=eval(line)
        for key in dic:
            rel_inherit.append((key,dic[key]))

with open("analyse/rel_use_class.txt", "r") as f_rel_use_class:
    content=f_rel_use_class.readlines()
    rel_use_class=[]
    for line in content:
        dic=eval(line)
        for key in dic:
            rel_use_class.append((key,dic[key]))

with open("analyse/rel_use_function.txt", "r") as f_rel_use_function:
    content=f_rel_use_function.readlines()
    rel_use_function=[]
    for line in content:
        dic=eval(line)
        for key in dic:
            rel_use_function.append((key,dic[key]))


handler = MedicalGraph()
handler.create_graphnodes()
handler.create_rel_use_function()
handler.creat_rel_inherit()
handler.creat_rel_class_variable()
handler.creat_rel_function_parameter()
handler.create_rel_use_class()
handler.create_rel_use()


'''#读取文件内容 r只读
f_functions=open("analyse/functions.txt","r")
f_class=open("analyse/class.txt","r")
f_parameter=open("analyse/parameter.txt","r")
f_codefile=open('analyse/codefile.txt',"r")
f_rel_use=open("analyse/rel_use.txt","r")
f_rel_class_variable=open("analyse/rel_class_variable.txt","r")
f_rel_function_parameter=open('analyse/rel_function_parameter.txt',"r")
f_rel_function_return=open('analyse/rel_function_return.txt',"r")
f_rel_herit=open('analyse/rel_inherit.txt','r')
f_rel_use_class=open('analyse/rel_use_class.txt','r')
f_rel_use_function=open('analyse/rel_use_function.txt','r')

#读取文件f_functions的内容，并将每行作为一个元素存储在一个名为content的列表中
content=f_functions.readlines()
#初始化一个名为functions的空列表，将在其中添加函数名
functions=[]
#启动一个for循环，遍历content列表中的每一行
#最终，functions列表将包含在functions.txt文件中定义的所有函数的名称
for line in content:
    #删除行首和行尾的任何空白字符(如换行符)，并将其添加到函数列表中。
    functions.append(line.strip())

content=f_class.readlines()
classes=[]
for line in content:
    classes.append(line.strip())

content=f_parameter.readlines()
parameters=[]
for line in content:
    parameters.append(line.strip())

content=f_codefile.readlines()
codefile=[]
for line in content:
    codefile.append(line.strip())


content=f_rel_use.readlines()
rels_use=[]
for line in content:
    #对于每一行，使用 eval 函数将其解析成一个字典，并将结果赋值给变量 dic
    dic=eval(line)
    #使用for循环遍历 dic 中的每个键值对，
    #对于每个键值对，将其转换成一个元组 (dic[key],key)，其中 key 是键，dic[key] 是值，
    #然后将该元组添加到 rels_use 列表中
    for key in dic:
        rels_use.append((dic[key],key))
    #输出结果：[('PythonProceduralTest.test', 'open'),....

content=f_rel_class_variable.readlines()
rel_class_variable=[]
for line in content:
    dic=eval(line)
    for key in dic:
        rel_class_variable.append((dic[key],key))

content=f_rel_function_parameter.readlines()
rel_function_parameter=[]
for line in content:
    dic=eval(line)
    for key in dic:
        rel_function_parameter.append((dic[key],key))

content=f_rel_function_return.readlines()
rel_function_return=[]
for line in content:
    dic=eval(line)
    for key in dic:
        rel_function_return.append((key,dic[key]))

content=f_rel_herit.readlines()
rel_herit=[]
for line in content:
    dic=eval(line)
    for key in dic:
        rel_herit.append((key,dic[key]))

content=f_rel_use_class.readlines()
rel_use_class=[]
for line in content:
    dic=eval(line)
    for key in dic:
        rel_use_class.append((key,dic[key]))

content=f_rel_use_function.readlines()
rel_use_function=[]
for line in content:
    dic=eval(line)
    for key in dic:
        rel_use_function.append((key,dic[key]))


    handler = MedicalGraph()
    handler.create_graphnodes()
    handler.create_rel_use_function()
    handler.creat_rel_inherit()
    handler.creat_rel_class_variable()
    handler.creat_rel_function_parameter()
    handler.create_rel_use_class()
    handler.create_rel_use()

'''