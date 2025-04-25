#这个特征也可以分一下，有从源码提取的特征以及字节码提取的特征是
#规则特征，做成one-hot向量，有几个就设成几维度的向量
#去加一个pattern5,.gas的限制有没有

#可扩充，因为补0变成了32维的向量
#输入代码片段
import re
from funs import extract_parameters
import pandas


#从智能合约代码中提取了几个预定义的安全模式，并将其结果编码为One-Hot向量
def extract_rule_feature(code,vectore_length): 
        pattern_num=4 #有几个pattern设几个[1,0]表示第1个pattern，[0,1]表示第2个pattern，如果存在这个pattern，就在后面追加1，不存在就0，变成[1,0,1]
        patterns = [[1 if i == j else 0 for j in range(pattern_num)] for i in range(pattern_num)]
        #---------------pattern1：call的参数在call之前是否有变量的变更,代码来自action1
        pattern = r'(.*)\.call\.value\s*(.*)'
        match = re.search(pattern, code)
        tmp=match[0]
        if 'bytes4' in tmp: 
                patterns[0].append(0)
        else:
                if "==" in match[0]:               
                        tmp=extract_parameters(match[0])[0]
                tmp=re.split("==",tmp)[0]
                tmp=re.split("=",tmp)[-1]      
                text=tmp
                if 'require' in text:
                        text=extract_parameters(tmp)[0]
                match2 = re.search(pattern, text)
                caller=match2.group(1)
                para=match2.group(2)
                para=extract_parameters(para)[0]
                code_lines = [line.strip() for line in code.split('\n') if line.strip()]
                flag0=0
                for i in range(len(code_lines)):
                        if '.call.value' in code_lines[i]:
                                flag0=i        
                flag1=False
                for i in range(flag0):
                        if ('-=' in code_lines[i] or '-' in code_lines[i] or '=' in code_lines[i]) and (caller in code_lines[i] or para in code_lines[i]):
                                flag1=True
                                break
                if flag1:
                        patterns[0].append(1)
                else:
                        patterns[0].append(0)

        #---------------pattern2：call的参数在call之后是否有变量的变更
                flag2=False
                for i in range(flag0,len(code_lines)):
                        if ('-=' in code_lines[i] or '-' in code_lines[i] or '=' in code_lines[i]) and (caller in code_lines[i] or para in code_lines[i]):
                                flag2=True
                                break
                if flag2:
                        patterns[1].append(1)
                else:
                        patterns[1].append(0)                

        #---------------pattern3：是否有lock机制，简化为是否有同一个变量先被设为true后被设为false
        code_lines = [line.strip() for line in code.split('\n') if line.strip()]
        # 查找变量被设置为 true 的操作
        locked_variables = set()
        for line in code_lines:
                if '=' in line and 'true' in line:
                        parts = line.split('=')
                        var_name = parts[0].strip()
                        locked_variables.add(var_name)
        # 检查是否存在将变量设置为 false 的操作
        lock_detected = False
        for line in code_lines:
                if '=' in line and 'false' in line:
                        parts = line.split('=')
                        var_name = parts[0].strip()
                        if var_name in locked_variables:
                                lock_detected = True
                                break

        if lock_detected:
                patterns[2].append(1)
        else:
                patterns[2].append(0)

        #---------------pattern4：require(address(pepeContract) == msg.sender);在.call之前限制了调用者
        code_lines = [line.strip() for line in code.split('\n') if line.strip()]
        flag3=0
        for i in range(len(code_lines)):
                if '.call.value' in code_lines[i]:
                        flag3=i        
        flag4=False
        for i in range(flag3):
                if 'require' in code_lines[i] and 'msg.sender' in code_lines[i]:
                        flag4=True
                        break
        if flag4:
                patterns[3].append(1)
        else:
            
                patterns[3].append(0)
     

        #---------------pattern5：查看有没有.gas的限制，以及里面的数字是多少



        #下面将没和特征补0扩充到32维
        extended_patterns = []
        for vector in patterns:
                extended_vector = vector + [0] * (vectore_length - len(vector))
                extended_patterns.append(extended_vector)
        return extended_patterns





#代码示例
code='''
function futrMiner() public nonReentrant  aaa payable {
require(msg.sender==123)
locked=true
require(futr.call.value(msg.value)());
a-=b;
uint256 mined = ERC20(futr).balanceOf(address(this));
ERC20(futr).approve(mny, mined);
MNY(mny).mine(futr, mined);
uint256 amount = ERC20(mny).balanceOf(address(this));
ERC20(mny).transfer(msg.sender, amount);
}


'''
# extract_rule_feature(code)
