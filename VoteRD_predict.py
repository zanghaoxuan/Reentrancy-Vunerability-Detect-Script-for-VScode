import re
import sys
import numpy as np
import json
import joblib
from tensorflow.keras.models import load_model
from models.y import RDvote
from funs import *
from feature.rule_feature import *
from feature.text_feature import *
from feature.two_position_feature import *

def predict(input_file_path):
    #TODO:load model
    model_dir = "./Save_Model"
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)

    modelX = load_model(f"{model_dir}/{config['modelX']}")
    modelY_1 = joblib.load(f"{model_dir}/{config['modelY_1']}")
    modelY_2 = joblib.load(f"{model_dir}/{config['modelY_2']}")
    rdvote = RDvote(modelX, modelY_1, modelY_2)

    #TODO: make input    
    rule_and_pos=[]
    with open(input_file_path, 'r', encoding = 'utf-8') as input_file:
        code = input_file.read()
        
    combine_snipples=[]
    #1 find contract
    contracts = re.split(r'(?:library|contract)\s+\w+\s*(?:is\s+\w+\s*)?{|(?:library|contract)\s+\w+\s+is\s+\w+,\s+\w+\s*', code)[1:] #因为还有contarct a is b,c的写法
    callcontract=""
    for contract in contracts:
        if '.call.value' in contract:
            callcontract=contract
            break
    if callcontract == "":
        print("[TRACE]No call function in this contract")
        return 0
    #2 find first .call in contract
    callfunction=""
    callfunction_array=[]
    fun_array=split_function(callcontract)
    fun_str=[['\n'.join(part)] for part in fun_array]
    for i in range(len(fun_str)):
        if '.call.value' in fun_str[i][0]:
            callfunction=fun_str[i][0]
            callfunction_array=fun_array[i]
            break
    if callfunction=="":    
        print("[ERROR]Please check your contract, your callback functions without name")
        return -1
    # #3 cross function
    pattern = r'function\s*(\w+)\s*\('  #func name
    match = re.search(pattern, callfunction_array[0])
    fun_name = match.group(1)    #name of .call func 

    #3 find func name
    pattern1 = r'function\s*(\w+)\s*\('     
    match = re.search(pattern1, callfunction)
    callname=match.group(1)

    #4 find modifier
    modifier_names=find_modifiers(code,callname)

    #5 merge func and modifier
    combine_snipple=callfunction
    for i in range(len(modifier_names)):
        pattern2=r'modifier\s+'+re.escape(modifier_names[i])+'\s*\(?.*\)?\s*{([^}]*)}'  #因为modifer一般比较简单，不会出现什么嵌套{}，所以就用这个匹配.但是()是可以不加的
        match=re.search(pattern2, code)    
        if match:         
            if "_;" not in match.group(1):  
                continue
            combine_snipple=match.group(1).replace("_;",combine_snipple)
        else:
            continue
    
   #ModelX-text
    vectore_length=60
    combine_snipples.append(combine_snipple)
    my_train_vector(combine_snipples,vector_length=vectore_length)
    text_feature = extract_text_feature(combine_snipple)
    
   #ModelY-position&rule
    #positional
    two_position_feature = extract_two_position_feature(combine_snipple,vectore_length) #position的简化版，用这个，不用solcx
    two_position_feature = np.array(two_position_feature)
    #rules
    rule_feature = extract_rule_feature(combine_snipple,vectore_length)
    rule_feature = np.array(rule_feature)
    #merge positon and rules
    combine_feature=np.concatenate([rule_feature, two_position_feature], axis=0)
    
    text = text_feature.reshape(1,100,60)
    rule_and_pos.append(combine_feature)
    rule_and_pos = np.stack(rule_and_pos)
    
    
    ans = rdvote.predict(text, rule_and_pos)
    return ans

if __name__ == '__main__':
    #input_file_name = sys.argv[1]
    #print("[INFO]Input File is: ", input_file_name)
    input_file_name = "./predict_input/tmp.sol"
    results = predict(input_file_name)
    if results == 1:
        print("Reentrancy vulnerability detected")
    else:
        print("No reentrancy issues found")
        
    
