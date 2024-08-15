import openai
import os
import numpy as np
import pandas as pd
import json
import io
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

#定义客户端
client = OpenAI(api_key=openai_api_key)

##调用测试
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role":"user","content":"什么是JSON Scheme？"}
#     ]
# )
# print(response.choices[0].message.content)
# print(response)

##定义函数
def sunwukong_function(data):
    """
    孙悟空算法函数，该函数定义了数据集计算过程
    :param data: 必要参数，表示带入计算的数据表，用字符串进行表示
    :return：sunwukong_function函数计算后的结果，返回结果为表示为JSON格式的Dataframe类型对象
    """
    data = io.StringIO(data)  ##将传入的字符串data转换成一个字符串流对象
    df_new = pd.read_csv(data,sep='\s+',index_col=0)
    res = df_new * 10
    return json.dumps(res.to_string())  

##创建一个DataFrame
df = pd.DataFrame({'x1':[1,2],'x2':[3,4]})

df_str = df.to_string()

data = io.StringIO(df_str)

df_new = pd.read_csv(data,sep='\s+',index_col=0)

##定义一个标准的Function Call函数
sunwukong = {
    "type":"function",
    "function":{"name":"sunwukong_function",
                "description":"用于执行孙悟空算法函数，定义了一种特殊的数据集计算过程",
                "parameters":{"type":"object",
                              "properties":{"data":{"type":"string",
                                                    "description":"执行孙悟空算法的数据集"},
                                            },
                               "required":["data"],
                             },         
                }

            }

# ##将函数放入工具列表
# tools = [sunwukong]



##取出注释信息
import inspect
# print(inspect.getdoc(sunwukong_function))
# print("-----------------------------------------")

##生成JSON Schema对象
# function_description = inspect.getdoc(sunwukong_function)
# response = client.chat.completions.create(
#     model="gpt-4-0613",
#     messages=[
#         {"role":"system","content":"以下是孙悟空函数的函数说明：%s" % function_description},
#         {"role":"user","content":"请帮我编写一个JSON Schema对象，用于说明孙悟空函数的参数输入规范。输出结果要求是JSON Schema格式的JONS类型对象，不需要任何前后修饰语句。"}
#     ]
# )

# print(response.choices[0].message.content)
# r = response.choices[0].message.content.replace("```","").replace("json","")
# print("-----------------------------------------")
# print(json.loads(r))
# print("-----------------------------------------")
# print(sunwukong)
# print("-----------------------------------------")
# print(sunwukong['function']['parameters'])
# function_name = "sunwukong_function"
# system_prompt = '以下是某的函数说明：%s' % function_description
# user_prompt = '根据这个函数的函数说明，请帮我创建一个JSON格式的字典，这个字典有如下5点要求：\
#                1.字典总共有三个键值对；\
#                2.第一个键值对的Key是字符串name，value是该函数的名字：%s，也是字符串；\
#                3.第二个键值对的Key是字符串description，value是该函数的函数的功能说明，也是字符串；\
#                4.第三个键值对的Key是字符串parameters，value是一个JSON Schema对象，用于说明该函数的参数输入规范。\
#                5.输出结果必须是一个JSON格式的字典，且不需要任何前后修饰语句' % function_name

# response = client.chat.completions.create(
#     model="gpt-4-0613",
#     messages=[
#         {"role":"system","content":system_prompt},
#         {"role":"user","content":user_prompt}
#     ]
# )
# json_function_description = json.loads(response.choices[0].message.content.replace("```","").replace("json",""))
# # print(json_function_description)
# json_str = {"type":"function","function":json_function_description}
# print(json_str)

def auto_functions(functions_list):
    """
    Chat模型的functions参数编写函数
    :param functions_list: 包含一个或者多个函数对象的列表；
    :return：满足Chat模型functions参数要求的functions对象
    """
    def functions_generate(functions_list):
        #创建空字典，用于保存每个函数的描述字典
        functions = []
        #对每个外部函数进行循环
        for function in functions_list:
            #读取函数对象的函数说明
            function_description = inspect.getdoc(function)
            #读取函数的函数名字符串
            function_name = function.__name__

            system_prompt = '以下是某的函数说明：%s' % function_description
            user_prompt = '根据这个函数的函数说明，请帮我创建一个JSON格式的字典，这个字典有如下5点要求：\
               1.字典总共有三个键值对；\
               2.第一个键值对的Key是字符串name，value是该函数的名字：%s，也是字符串；\
               3.第二个键值对的Key是字符串description，value是该函数的函数的功能说明，也是字符串；\
               4.第三个键值对的Key是字符串parameters，value是一个JSON Schema对象，用于说明该函数的参数输入规范。\
               5.输出结果必须是一个JSON格式的字典，且不需要任何前后修饰语句' % function_name
            
            response = client.chat.completions.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ]
                )
            json_function_description = json.loads(response.choices[0].message.content.replace("```","").replace("json",""))
            json_str={"type": "function","function":json_function_description}
            functions.append(json_str)
        return functions
    ##最大可以尝试4次
    max_attempts=4
    attempts = 0

    while attempts<max_attempts:
        try:
            functions = functions_generate(functions_list)
            break
        except Exception as e:
            attempts+=1
            print("发生错误:",e)
            if attempts == max_attempts:
                print("已达到最大尝试次数，程序终止。")
                raise #重新引发最后一个异常
            else:
                print("正在重新运行...")
    return functions

# functions_list = [sunwukong_function]
# tools = auto_functions(functions_list)
# print(tools)


# print(response.choices[0].message)

def tangseng_function(data):
    """
    唐僧算法函数，该函数定义了数据集计算过程
    :param data: 必要参数，表示带入计算的数据表，用字符串进行表示
    :return：tangseng_function函数计算后的结果，返回结果为表示为JSON格式的Dataframe类型对象
    """
    data = io.StringIO(data)
    df_new = pd.read_csv(data,sep='\s+',index_col=0)
    res = df_new * 1000000
    return json.dumps(res.to_string())

functions_list = [sunwukong_function,tangseng_function]
tools = auto_functions(functions_list)

print(tools)

##定义工具函数字典
available_tools = {
    "sunwukong_function":sunwukong_function,
    "tangseng_function":tangseng_function,
}

messages = [
    {"role":"system","content":"数据集data: %s,数据集以字符串的形式呈现" % df_str},
    {"role":"user","content":"请在数据集data上执行唐僧算法"}
]
response = client.chat.completions.create(
    model="gpt-4-0613",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(response.choices[0].message)
print("*******************************")
print(response)

for tool_call in response.choices[0].message.tool_calls:
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    #zhixing
    results = available_tools[function_name](**arguments)
    results_str = json.loads(results)
    print(results_str)