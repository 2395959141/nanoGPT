import gradio as gr
import torch
from transformers import BertTokenizer
from m_model import GPT, GPTConfig

# 配置参数
MODEL_PATH = "/path/to/your/model_checkpoint.pt"  # 请替换为实际模型路径
TOKENIZER_PATH = "/root/autodl-fs/huggingface/transformers/bert_base_chinese_tokenicker"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512  # 与prepare.py中的block_size一致

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# 加载模型
def load_model():
    # 初始化配置（参数需要与训练时一致）
    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=MAX_LENGTH,
        vocab_size=tokenizer.vocab_size
    )
    
    # 创建模型实例
    model = GPT(config)
    
    # 加载checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# 生成函数
def generate_text(prompt, temperature=0.7, top_k=50):
    try:
        # 编码输入
        input_ids = tokenizer.encode(
            prompt,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            truncation=True
        )
        
        # 转换为tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        
        # 生成文本
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_new_tokens=100,  # 每次生成100个token
                temperature=temperature,
                top_k=top_k
            )
        
        # 解码输出
        output_ids = generated[0].cpu().tolist()
        decoded_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # 高亮新生成部分
        original_length = len(input_ids)
        return f"{prompt}<span style='color: #2ecc71'>{decoded_text[len(prompt):]}</span>"
    
    except Exception as e:
        return f"生成错误：{str(e)}"

# 创建Gradio界面
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, label="输入文本", placeholder="请输入你的提示词..."),
        gr.Slider(0.1, 2.0, value=0.7, label="温度（控制随机性）"),
        gr.Slider(1, 100, value=50, step=1, label="Top-k（候选词数量）")
    ],
    outputs=gr.HTML(label="生成结果"),
    title="中文文本生成演示",
    examples=[
        ["从前有座山，山里有座庙，庙里有个和尚说："],
        ["人工智能的未来发展将会"]
    ],
    css=".gradio-container {background: #f5f6fa}"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
