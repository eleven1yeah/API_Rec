import os
import json
from py2neo import Graph,Node

functions=["test1","test2"]
rels_use=[("test1","test2")]

class APIGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.g = Graph("http://localhost:7474", auth=("xxx", "xxx"))

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

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
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

with open("analyse/functions.txt", "r") as f_functions:
    content = f_functions.readlines()
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
        dic = eval(line)
        for key in dic:
            rels_use.append((dic[key], key))

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


handler = APIGraph()
handler.create_graphnodes()
handler.create_rel_use_function()
handler.creat_rel_inherit()
handler.creat_rel_class_variable()
handler.creat_rel_function_parameter()
handler.create_rel_use_class()
handler.create_rel_use()


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

content=f_functions.readlines()
functions=[]
for line in content:
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
    dic=eval(line)
    for key in dic:
        rels_use.append((dic[key],key))

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


    handler = APIGraph()
    handler.create_graphnodes()
    handler.create_rel_use_function()
    handler.creat_rel_inherit()
    handler.creat_rel_class_variable()
    handler.creat_rel_function_parameter()
    handler.create_rel_use_class()
    handler.create_rel_use()

'''
