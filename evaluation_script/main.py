import random
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import requests
import os
lock = Lock()
API_KEY_vqa = "0c9745f6e0254f41818839057a62025b.EZ6Cq8FgHRPo91fk"
MODEL_vqa = "glm-4-flash"  # 可用的模型: glm-3-turbo, glm-4, characterglm
def call_zhipuai_api(api_key, model, messages, temperature=0.0, max_tokens=1024):
    """
    调用智谱AI API (最新版本)
    
    参数:
    api_key: 智谱API密钥 (格式: your_api_key)
    model: 模型名称 (如: "glm-4")
    messages: 对话消息列表 [{"role": "user", "content": "你好"}]
    """
    # 1. API端点
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    # 2. 准备请求头
    headers = {
        "Authorization": f"Bearer {api_key}",  # 直接使用API Key
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # 3. 准备请求体
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # 4. 发送请求
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=300
        )
        
        # 5. 处理响应
        if response.status_code == 200:
            result = response.json()
            
            # 提取响应内容
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"API响应格式异常: {json.dumps(result, indent=2)}"
        
        # 处理错误响应
        error_info = f"HTTP {response.status_code} 错误"
        try:
            error_data = response.json()
            if "error" in error_data:
                error_info += f": {error_data['error']['message']}"
            elif "msg" in error_data:
                error_info += f": {error_data['msg']}"
        except:
            error_info += f"\n响应文本: {response.text[:200]}"
        
        return f"❌ {error_info}"
    
    except requests.exceptions.RequestException as e:
        return f"❌ 请求失败: {str(e)}"

def process_line(data,counter_dict):
    question = data['question']
    
    # 构造请求
    try:
        # 创建对话
        messages=[
                {"role": "system", "content": "你是一个智能判断助手，你的任务是根据问题和给出的正确答案，分析模型的预测答案是是否正确回答了问题。如果正确，请回答1；如果错误，请回答0；注意，只能输出0或者1，不允许有任何其他内容输出。正确的预测必须完全符合标准答案的语义，否则被认为错误"},
                {"role": "user", "content": f"问题是{question};模型的预测答案是：{data['predict']}；正确答案是：{data['label']}"}
            ]
        print("正在调用智谱AI API...")
        # result =[]
        # for i in range(5):
        is_true = call_zhipuai_api(API_KEY_vqa, MODEL_vqa, messages)
        # result.append(is_true)
        # is_true = find_most_frequent(result)
        print(f"API返回: {is_true}")
    except Exception as e:
        print(f"API请求失败: {e}")
        is_true = "0"
    # 线程安全写入
    with lock:
        content = {
            "question": question,
            "predict": data["predict"],
            "label": data["label"],
            "is_true": is_true,
            "image":data["images"] if "images" in data.keys() else '',
        }
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(content, ensure_ascii=False) + '\n')
        
        # 更新计数器
        if "1" == str(is_true):
            counter_dict['true_counter'] += 1
        else:
            print(f"错误的预测: {data['predict']}，正确答案: {data['label']},istrue: {is_true},image: {data['image']},question: {data['question']}")
        counter_dict['count'] += 1
        
def compute_iou(box1, box2):
    """
    计算两个边界框的IOU（Intersection over Union）
    :param box1: 第一个边界框 [[x1, y1], [x2, y2]]
    :param box2: 第二个边界框 [[x1, y1], [x2, y2]]
    :return: IOU值
    """
    # 标准化边界框坐标（确保左上角和右下角）
    x1s = [min(p[0] for p in box1), min(p[0] for p in box2)]
    x2s = [max(p[0] for p in box1), max(p[0] for p in box2)]
    y1s = [min(p[1] for p in box1), min(p[1] for p in box2)]
    y2s = [max(p[1] for p in box1), max(p[1] for p in box2)]

    # 获取交集坐标
    inter_x1 = max(x1s[0], x1s[1])
    inter_y1 = max(y1s[0], y1s[1])
    inter_x2 = min(x2s[0], x2s[1])
    inter_y2 = min(y2s[0], y2s[1])

    # 计算交集面积
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # 计算各自面积
    area1 = (x2s[0] - x1s[0]) * (y2s[0] - y1s[0])
    area2 = (x2s[1] - x1s[1]) * (y2s[1] - y1s[1])

    # 计算并集面积
    union_area = area1 + area2 - inter_area

    # 计算IOU
    return inter_area / union_area if union_area > 0 else 0.0

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    output = {}
    if phase_codename == "VG-RS":
        print("Evaluating for VG-RS Phase")
        with open(test_annotation_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        with open(user_submission_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
            # 构建用户提交数据的查找字典
        user_dict = {(item['image_path'], item['question']): item for item in user_data}
        # 存储结果
        accum_acc = 0
        # 遍历测试数据
        for item in test_data:
            key = (item['image_path'], item['question'])
            if key in user_dict:
                # 获取两个result并计算IOU
                box1 = item['result']
                box2 = user_dict[key]['result']
                iou = compute_iou(box1, box2)
                if iou >= 0.5:
                    accum_acc += 1
            else:
                output["result"] = [{"test_split": {"ACC": NAN}}]
                output["submission_result"] = output["result"][0]
                return output
        accum_acc = accum_acc / len(test_data)
        output["result"] = [
            {
                "test_split": {
                    "ACC": accum_acc,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for VG-RS Phase")
    elif phase_codename == "VQA-SA":
        print("Evaluating for VQA-SA Phase")
        with open(test_annotation_file, 'r', encoding='utf-8') as f:
            result_new_data = json.load(f)  # 包含label的数据
        with open(user_submission_file, 'r', encoding='utf-8') as f:
            new_data = json.load(f)  # 包含predict的数据
        # 创建索引字典：键为(image_path, question)，值为result
        indexed_new_data = defaultdict(dict)
        for item in new_data:
            key = (item["image_path"], item["question"])
            indexed_new_data[key] = item["result"]
    
        # 创建临时数据结构
        temp_data = []
        counters = {'true_counter': 0, 'count': 0}
        # 遍历result_new_data中的每个条目
        for item in result_new_data:
            key = (item["image_path"], item["question"])
            # 获取label（来自3dsr_result_new.json）
            label = item["result"]
            # 获取predict（来自3dsr_new.json）
            predict = indexed_new_data.get(key)
            # 如果找到匹配项，添加到临时数据
            if predict is not None:
                temp_data.append({
                    "question": item["question"],
                    "predict": predict,
                    "label": label,
                    "image": item["image_path"]
                })
            else:
                output["result"] = [{"test_split": {"ACC": NAN}}]
                output["submission_result"] = output["result"][0]
                return output
                
        from functools import partial
        task_func = partial(
            process_line,
            counter_dict=counters
        )
        with ThreadPoolExecutor(max_workers=180) as executor:  # 根据API限制调整线程数
            executor.map(task_func, temp_data)
        
        output["result"] = [
            {
                "test_split": {
                    "ACC1":str(counter_dict['true_counter'])
                    "ACC2":str(counter_dict['count'])
                    # "ACC": counters['true_counter'] / counters['count'] if counters['count'] else 0,
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for VQA-SA Phase")
    elif phase_codename == "VR-Ads":
        print("Evaluating for VR-Ads Phase")
        output["result"] = [
            {
                "test_split": {
                    "Metric1": random.randint(0, 99),
                    "Metric2": random.randint(0, 99),
                    "Metric3": random.randint(0, 99),
                    "Total": random.randint(0, 99),
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for VR-Ads Phase")
    return output
