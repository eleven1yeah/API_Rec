import ast

class CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
        self.current = None

    def visit_Call(self, node):
        # new call, trace the function expression
        self.current = ''
        self.visit(node.func)
        self.calls.append(self.current)
        self.current = None

    def generic_visit(self, node):
        if self.current is not None:
            print("warning: {} node in function expression not supported".format(
                  node.__class__.__name__))
        super(CallCollector, self).generic_visit(node)

    # record the func expression 
    def visit_Name(self, node):
        if self.current is None:
            return
        self.current += node.id

    def visit_Attribute(self, node):
        if self.current is None:
            self.generic_visit(node)
        self.visit(node.value)  
        self.current += '.' + node.attr

tree = ast.parse(yoursource)
cc = CallCollector()
cc.visit(tree)
print(cc.calls)