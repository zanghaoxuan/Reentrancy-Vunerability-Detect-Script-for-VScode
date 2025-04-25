from solidity_parser import parser


def split_function(contract): #输入文件名，分割函数
    function_list = []
    lines = contract.split('\n')
    flag = -1  # 作为记号
    for line in lines:
        text = line.strip()
        if len(text) > 0 and text != "\n":        
            if text.split()[0] == "function":
                function_list.append([text])
                flag += 1
            elif len(function_list) > 0 and ("function" in function_list[flag][0]):
                function_list[flag].append(text)
    return function_list


def find_modifiers(code,function_name):
    ast = parser.parse(code)                    #获取solidity版本
    modifiers=[]
    for node in ast['children']:
        # 检查是否是合约定义节点
        if node.get('type') == 'ContractDefinition':
            # 遍历合约中的每个子节点
            for sub_node in node.get('subNodes', []):
                # 检查是否是函数定义节点，并且函数名匹配
                if sub_node.get('type') == 'FunctionDefinition' and sub_node.get('name') == function_name:
                    modifier_invocations = sub_node.get('modifiers', [])
    modifier_names = [modifier['name'] for modifier in modifier_invocations]
    return modifier_names


def extract_parameters(code): #输入一句代码，找到遇到的第一个()里面的内容
    parameters = []
    stack = []
    current_parameter = ''
    for char in code:
        if char == '(':
            if stack:
                current_parameter += char
            stack.append('(')
        elif char == ')':
            if stack:
                stack.pop()
                if stack:
                    current_parameter += char
                else:
                    parameters.append(current_parameter)
                    current_parameter = ''
        elif stack:
            current_parameter += char

    return parameters


# solcx.install_solc("0.4.25")
# solcx.set_solc_version("0.4.25")
# compiled_sol = solcx.compile_source(code)
# contract_names = list(compiled_sol.keys()) 