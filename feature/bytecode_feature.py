#那篇文章将全文的bytecode拿去做图片，显然是没有意义的，我们只找到关键函数的bytecode
#提取关键函数的bytecode，预处理方便后面提取特征
#文献34

import  solcx 
from solidity_parser import parser
import os
import numpy as np

#找到函数所在的合约的名字，从sol文件中解析出来AST
def find_contract_with_function(sol_file_path, function_name): 
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

#编译指定的合约并提取其字节码。
def extract_bytecode_feature(file,code,callname):
    #先找到合约的名字，只要这个合约的hytecode
    call_contract_name=find_contract_with_function(file,callname)
    ast = parser.parse(code)
    solidity_version=''
    for node in ast['children']:
        if node.get('type') == 'PragmaDirective':
            version=node['value'].replace('^','')
            print(version)
            if version < '0.4.11':
                solidity_version='0.4.11'
            else:    
                solidity_version=version
            break
    print(solidity_version)
    solcx.install_solc(solidity_version)
    solcx.set_solc_version(solidity_version)
    compiled_sol = solcx.compile_source(code)
    contract_names = list(compiled_sol.keys())  #可以得到合约的名字
    for target_contract in contract_names:
        if call_contract_name in target_contract:
            call_contract_name=target_contract
            break
      
    contract_bytecode = compiled_sol[call_contract_name]['bin']
    return contract_bytecode


# byte_values = [int(bytecode[i:i+2], 16) for i in range(0, len(bytecode), 2)]

# # 计算图像的高度
# height = len(byte_values) // 100
# if len(byte_values) % 100 != 0:
#     height += 1

# padding_length = height * 100 - len(byte_values)

# # 将不够的部分补0
# byte_values += [0] * padding_length

# # 塑形为 n x 100 的矩阵
# image_matrix = np.array(byte_values).reshape(height, 100)
# # 创建图像对象
# print(image_matrix.shape)

# exit()









# # 提取合约字节码
# contract_bytecode2 = compiled_sol['<stdin>:aaa']['bin']  #这个和[bin]的区别是什么
# print(contract_bytecode2)
# contract_abi=compiled_sol['<stdin>:aaa']['abi']

#     # function setData(uint _data) public {
#     #     data = _data;
#     # }