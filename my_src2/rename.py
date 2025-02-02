import os

# 指定要处理的目录
directory = "/root/autodl-fs/huggingface/transformers/bert_base_chinese_tokenizer"

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 去掉空格和括号
    new_name = filename.replace(" ", "").replace("(", "").replace(")", "").replace("1", "")
    
    # 如果文件名有变化，则重命名
    if new_name != filename:
        # 获取完整路径
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_name}")