#提取代码中使用的函数和API调用
import ast
import os

rel_uses = []
name = ""
functions=set()
classes=set()
rel_inherit={}
rel_function_parameters=[]
rel_class_variables=[]
rel_function_returns={}
codefiles=set()
parameters=set()
class CallCollector(ast.NodeVisitor): #遍历抽象语法树
    def __init__(self):
        self.calls = []
        self._current = []
        self.args = []
        self.assigns = []
        self._in_call = False
        self._in_class = False
        self._in_init =False
        self.father=""
        self._in_assign= False
        self._is_selfassign =False
        self._return=""
        self.classname=""
    def visit_Call(self, node):
        self._current = []
        self._in_call = True
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self._in_call:
            self._current.append(node.attr)
        self.generic_visit(node)
        if self._is_selfassign:
            self.assigns.append(node.attr)
            #print((node.attr))
            self._is_selfassign= False


    def visit_Name(self, node):
        if self._in_call:
            self._current.append(node.id)
            self.calls.append('.'.join(self._current[::-1]))
            # Reset the state
            self._current = []
            self._in_call = False
        if self._in_assign and node.id=="self":
            self._is_selfassign= True
        elif self._in_class:
            self.father=node.id
            self._in_class= False
        self.generic_visit(node)
    def visit_FunctionDef(self, node):
        global functions,rel_uses,classes,rel_inherit,rel_function_parameters,rel_function_returns
        self.args=[]
        name=self.classname+"."+node.name
        if name=="self":
            name=self.classname
        functions.add(name)
        #print(node.name)
        type(node).__name__
        if name=="__init__":
            self._in_init=True
        ast.NodeVisitor.generic_visit(self, node)
        self._in_init = False
        dic1={}
        print(name+"的参数有"+",".join(self.args))
        for arg in self.args:
            parameters.add(arg)

        for arg in self.args:
            dic1[arg] = name
        rel_function_parameters.append(dic1)
        dic={}
        print(name+"调用的函数有"+",".join(self.calls))
        for function in self.calls:
            dic[function]=name
        rel_uses.append(dic)
        print(name+"返回值为："+self._return)
        rel_function_returns[name]=self._return
        #print(dic)
        #fp2.write(str(dic)+'\n')
        self.calls=[]


    def visit_ClassDef(self, node):
        global functions, rel_uses, classes, rel_inherit, rel_function_parameters, rel_function_returns
        self.classname=node.name
        print("类名为："+node.name)
        classes.add(node.name)
        self._in_class= True
        self.generic_visit(node)
        print(node.name+"的变量有"+",".join(self.assigns))
        for assign in self.assigns:
            parameters.add(assign)
        dic = {}
        for paramter in self.assigns:
            dic[paramter] = node.name
        rel_class_variables.append(dic)
        self.assigns = []
        print(node.name+"的父类是："+self.father)
        rel_inherit[node.name]=self.father
        classes.add(self.father)
        self.father=""
    def visit_arg(self, node) :
        if node.arg=="self":
            arg=self.classname
        else:
            arg=node.arg
        self.args.append(arg)
        self.generic_visit(node)
    def visit_Assign(self, node):
        self._in_assign= True
        self.generic_visit(node)
        self._in_assign= False
    def visit_Return(self, node) :
        self._return=ast.dump(node.value)
        self.generic_visit(node)

def analyse():
    fillist = os.listdir("code")


    for filepath in fillist:
        print("==" + filepath + "==")
        f = open("code/" + filepath, 'r')
        try:
            tree = ast.parse(f.read())
            cc = CallCollector()
            cc.visit(tree)
            filename='_'.join(filepath.split('_')[3:])
            global functions,rel_uses,classes,rel_inherit,rel_function_parameters,rel_function_returns
            global fp1,fp2,fp3,fp4,fp5,fp6,fp7,fp8,fp9,fp10
            dic1={}
            dic2={}
            for word in functions:
                fp1.write(word+'\n')
                dic1[word]=filename
            functions.clear()
            for dic in rel_uses:
                fp2.write(str(dic)+'\n')
            for word in classes:
                fp3.write(word+'\n')
                dic2[word] = filename
            fp4.write(str(rel_inherit)+'\n')
            rel_inherit={}
            for dic in rel_function_parameters:
                fp5.write(str(dic)+'\n')
            for dic in rel_class_variables:
                fp6.write(str(dic)+'\n')
            fp7.write(str(rel_function_returns)+'\n')
            fp8.write(filename+'\n')
            fp9.write(str(dic1)+'\n')
            fp10.write(str(dic2) + '\n')
            for word in parameters:
                fp11.write(word + '\n')
            #print(cc.calls)

        except Exception as re:
            print("+++error++++")
            print(re)


''

"""tree = ast.parse('''
def visit_Name(self, node):
    if self._in_call:
        self._current.append(node.id)
        self.calls.append('.'.join(self._current[::-1]))
        # Reset the state
        self._current = []
        self._in_call = False
    self.generic_visit(node)
def visit_Attribute(self, node):
    if self._in_call:
        self._current.append(node.attr)
    self.generic_visit(node)
 ''')"""
fp1 = open('analyse/' + "functions" + '.txt', 'w', encoding='utf-8')
fp2 = open('analyse/' + "rel_use" + '.txt', 'w', encoding='utf-8')
fp3 = open('analyse/' + "class" + '.txt', 'w', encoding='utf-8')
fp4 = open('analyse/' + "rel_inherit" + '.txt', 'w', encoding='utf-8')
fp5 = open('analyse/' + "rel_function_parameter" + '.txt', 'w', encoding='utf-8')
fp6 = open('analyse/' + "rel_class_variable" + '.txt', 'w', encoding='utf-8')
fp7 = open('analyse/' + "rel_function_return" + '.txt', 'w', encoding='utf-8')
fp8 = open('analyse/' + "codefile" + '.txt', 'w', encoding='utf-8')
fp9 = open('analyse/' + "rel_use_function" + '.txt', 'w', encoding='utf-8')
fp10 = open('analyse/' + "rel_use_class" + '.txt', 'w', encoding='utf-8')
fp11 = open('analyse/' + "parameter" + '.txt', 'w', encoding='utf-8')
analyse()
print("finish")
