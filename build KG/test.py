import ast

dic = {}
name = ""
functions=set()
class CallCollector(ast.NodeVisitor):
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
        global name
        global fp1,fp2
        global functions
        self.args=[]
        name=self.classname+"."+node.name
        if name=="self":
            name=self.classname
        functions.add(node.name)
        #print(node.name)
        type(node).__name__
        if name=="__init__":
            self._in_init=True
        ast.NodeVisitor.generic_visit(self, node)
        self._in_init = False

        print(name+"的参数有"+",".join(self.args))
        global dic
        print(name+"调用的函数有"+",".join(self.calls))
        for func in self.calls:
            dic[func]=name
            functions.add(func)
        print(name+"返回值为："+self._return)
        #print(dic)
        #fp2.write(str(dic)+'\n')
        self.calls=[]
        dic={}
    def visit_ClassDef(self, node):
        self.classname=node.name
        print("类名为："+node.name)
        self._in_class= True
        self.generic_visit(node)
        print(node.name+"的变量有"+",".join(self.assigns))
        self.assigns = []
        print(node.name+"的父类是："+self.father)
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






class Father(object):
    def __init__(self, name):
        self.name=name
        print ( "name: %s" %( self.name) )
    def getName(self):
        return 'Father ' + self.name

root_node=ast.parse("""
class Father(object):
    def __init__(self, name):
        self.bame=name
        print ( "name: %s" %( self.name) )
        self.aame=name
        name_=name
    def getName(self):
        return 'Father ' + self.name
""")
print(ast.dump(root_node))
cc=CallCollector()
cc.visit(root_node)