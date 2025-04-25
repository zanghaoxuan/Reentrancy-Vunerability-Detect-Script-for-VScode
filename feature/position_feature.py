#尽量把所有样本的版本都设置一样的，不然编译结果不一样容易报错
#从solidity合约中提取位置信息并对其标准化处理并返回100维向量
import solcx
from solidity_parser import parser
import os
import numpy as np
import pprint
from sklearn.preprocessing import MinMaxScaler

#AST节点的提取函数
#递归遍历AST节点，提取Assignment和call节点的源位置，将这些位置存储在assignment_src和call_src列表中，并计数Assignment节点的数量
def extract_src_from_ast(ast_node,assignment_num,call_src,assignment_src):
    if isinstance(ast_node, dict):
        if 'nodeType' in ast_node and ast_node['nodeType'] == 'Assignment':

            assignment_num[0] += 1 
            assignment_src.append(int(ast_node['src'].split(":")[0]))

        # if 'src' in ast_node and ast_node['nodeType']=="Assignment":
        #     print(1)
        #     print("Found assignment at src:",ast_node['src'])
            
        if 'nodeType' in ast_node and ast_node['nodeType'] == 'MemberAccess' and ast_node.get('memberName') == 'call':

            call_src.append(int(ast_node['src'].split(":")[0]))


        for key, value in ast_node.items():
            if isinstance(value, (dict, list)):
                extract_src_from_ast(value,assignment_num,call_src,assignment_src)
    elif isinstance(ast_node, list):
        for item in ast_node:
            extract_src_from_ast(item,assignment_num,call_src,assignment_src)

#寻找特定函数
#递归遍历AST，寻找特定名称的函数，并返回该函数的AST节点
def find_function(ast_node, function_name):

    if ast_node['nodeType'] == 'FunctionDefinition' and ast_node['name'] == function_name:
        return ast_node
    if 'nodes' in ast_node:
        for child_node in ast_node['nodes']:
            result = find_function(child_node, function_name)
            if result:
                return result
    return None

#找到包含特定函数的合约，遍历合约，找到包含特定函数的合约名称
def find_contract_with_function(sol_file_path, function_name): #找到函数所在的合约的名字
    with open(sol_file_path, 'r') as f:
        # 读取.sol文件内容
        sol_code = f.read()

    # 解析.sol文件
    ast = parser.parse(sol_code)
    # 遍历合约列表
    for contract in ast['children']:
        if contract['name'] and contract['type'] == 'ContractDefinition':
            # 检查合约中的函数列表
            for item in contract['subNodes']:
                if item['type'] == 'FunctionDefinition' and item['name'] == function_name:
                    return contract['name']

    return None

#提取位置特征的主函数
def extract_position_feature(file,code,callname):
    #找到包含特定函数的合约名称
    call_contract_name=find_contract_with_function(file,callname)
    ast = parser.parse(code)
    
    #提取pragma指令确定solidity版本
    solidity_version=''
    for node in ast['children']:
        if node.get('type') == 'PragmaDirective':
            version=node['value'].replace('^','')
            if version.split('.')[1] == '4' and version.split('.')[2]<'18':
                solidity_version='0.4.25'
            else:    
                solidity_version=version
            break

    #安装并设置Solidity编译器版本
    solcx.install_solc(solidity_version)
    solcx.set_solc_version(solidity_version)
    #编译Solidity代码并提取合约的AST
    compiled_sol = solcx.compile_source(code)
    contract_names = list(compiled_sol.keys())  #可以得到合约的名字
    for target_contract in contract_names:
        if call_contract_name in target_contract:
            call_contract_name=target_contract
            break
    #从合约AST中找到特定函数的AST
    contract_ast = compiled_sol[call_contract_name]['ast']
    call_ast = find_function(contract_ast, callname)
    assignment_num = [0] #注意这里如果写成assignment_num=0的话，传入参数之后输出并不会变化
    call_src=[]
    assignment_src=[]
    #提取函数体中的Assignment和call节点的位置，并计数Assignment节点
    extract_src_from_ast(call_ast['body'],assignment_num,call_src,assignment_src)
    #合并提取的特征，填充为100维度的列表
    merged_list = call_src + assignment_num + assignment_src + [0] * (100 - len(call_src) - len(assignment_num) - len(assignment_src))
    #下面我把数组的范围变成了-1-1
    min_val = np.min(merged_list)
    max_val = np.max(merged_list)
    scaled_array = -1 + 2 * (merged_list - min_val) / (max_val - min_val)
    return [scaled_array]  #100维度




# with open('dataset/code/simple_dao.sol') as f:
#     code=f.read()
# extract_position_feature('dataset/code/simple_dao.sol',code,'withdraw')





