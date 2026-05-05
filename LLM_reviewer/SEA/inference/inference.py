from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
import torch
model_name = "ECNU-SEA/SEA-E"
cache_path = '/mnt/duyna/review_assessment/model/SEA'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)

chat_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir=cache_path,
    torch_dtype=torch.float16, 
    device_map="auto"
)



def read_txt_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def read_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_output(output, save_dir, data_id):
    with open(save_dir + data_id + ".txt", 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"The summary review of paper {data_id} has been saved.")
    f.close()

def get_subfile(path):
    subfiles = [d for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))]
    return subfiles

def infer_one(mmd_file_path):
    system_prompt_dict = read_json_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),"template.json"))
    instruction = system_prompt_dict['instruction_e']
    paper = read_txt_file(mmd_file_path)
    idx = paper.find("## References")
    paper = paper[:idx].strip()
    # prompt = instruction + '\n' + paper
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": paper},
    ]
    encodes = tokenizer.apply_chat_template(messages, return_tensors="pt")
    encodes = encodes.to(chat_model.device)
    len_input = encodes.shape[1]
    generated_ids = chat_model.generate(encodes,max_new_tokens=8192,do_sample=True)
    response = tokenizer.batch_decode(generated_ids[: , len_input:])[0]
    return response

def run_review(mmd_file_path):
    infer_modelname = model_name.split('/')[-2]
    infer_save_path = "./" + infer_modelname + '/'
    print(infer_modelname)
    if not os.path.exists(infer_save_path):
        os.mkdir(infer_save_path)
    res = infer_one(mmd_file_path)
    return res


if __name__ == "__main__":
    # 1. Đường dẫn thư mục Input và Output
    input_dir = "/mnt/duyna/review_assessment/data/extracted_files_500_iclr2024/papers_mmd" 
    output_dir = "/mnt/duyna/review_assessment/sea_output_iclr2024"
    
    # Tự động tạo thư mục output nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not output_dir.endswith('/'):
        output_dir += '/'
    
    # 2. Lấy danh sách các file trong thư mục
    files = get_subfile(input_dir)
    
    # 3. Lặp qua từng file để review
    for file_name in tqdm(files, desc="Đang review các papers"):
        if file_name.endswith(".mmd"):
            data_id = file_name.replace(".mmd", "")
            
            # --- TÍNH NĂNG CHECK OUTPUT (BỎ QUA FILE ĐÃ CHẠY) ---
            out_file_path = os.path.join(output_dir, data_id + ".txt")
            if os.path.exists(out_file_path):
                # Đã có file kết quả -> Bỏ qua, chạy luôn vòng lặp tiếp theo
                continue
            # ----------------------------------------------------
            
            file_path = os.path.join(input_dir, file_name)
            
            # Chạy inference trực tiếp
            review = infer_one(file_path)
            
            # Lưu file txt kết quả
            save_output(review, output_dir, data_id)
