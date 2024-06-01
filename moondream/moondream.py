import os
import sys
sys.path.insert(1,os.path.abspath('./moondream/transformer4391'))
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformer4391.transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import torch

def describe():
    image_path = "./keyframe_extract/extract_result/keyframe_943.jpg"
    model_id = "vikhyatk/moondream2"
    revision = "2024-03-13"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    T1 = time.time()

    # image = Image.open(image_path)
    # enc_image = model.encode_image(image)
    # print(model.answer_question(enc_image, "Describe this image.", tokenizer))

    T2 = time.time()
    print('程序运行时间:%s秒' % ((T2 - T1)))

    answers = model.batch_answer(
        images=[Image.open(image_path), Image.open(image_path)],
        prompts=["Describe this image.", "Describe this image."],
        tokenizer=tokenizer,
    )
    print(answers)

    T3 = time.time()
    print('程序运行时间:%s秒' % ((T3 - T2)))


if __name__=="__main__":
    model_id = "vikhyatk/moondream2"
    revision = "2024-03-13"
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, trust_remote_code=True, revision=revision
    # )
    model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16, 
    # attn_implementation="flash_attention_2"
).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    # Directory containing images
    directory = 'keyframe_extract\\extract_result'


    # 读取命令行参数
    if len(sys.argv) > 2:
        total_caption_num = int(sys.argv[1])*2
        fps = int(sys.argv[2])
    else:
        total_caption_num = 20  # 如果没有提供参数，使用默认值10
        fps = 30

    # Get all files in the directory
    files = os.listdir(directory)

    # Filter out image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Custom sorting function, sort by numbers in filename
    def sort_key(filename):
        parts = filename.split('_')
        number_part = parts[1].split('.')[0]
        return int(number_part)

    # Sort files by number
    sorted_image_files = sorted(image_files, key=sort_key)


    # Calculate step to use based on desired number of captions
    if len(sorted_image_files) > total_caption_num:
        step = len(sorted_image_files) // total_caption_num
    else:
        step = 1  # Use all available images if less than requested

    # Select images based on calculated step
    sorted_image_files = sorted_image_files[::step][:total_caption_num]

    # Process images in batches of three
    batch_size = 3
    selected_images_batches = [sorted_image_files[i:i + batch_size] for i in range(0, len(sorted_image_files), batch_size)]

    # Using the model's batch processing capability
    T1 = time.time()

    # Initialize results list


    for batch_index, batch in enumerate(selected_images_batches):
        images = [Image.open(os.path.join(directory, image_file)) for image_file in batch]
        prompts = ["Briefly describe the image." for _ in images]
        answers = model.batch_answer(images=images, prompts=prompts, tokenizer=tokenizer)
        
        # Collect answers and format them
        for i, answer in enumerate(answers):
            frame_number = int(batch[i].split('_')[-1].split('.')[0])
            second = round((frame_number - 1) / fps)
            result_text = f"Second {second}: {answer}"
            print(result_text)

    T2 = time.time()

