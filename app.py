from keyframe_extract import keyframes_extract_diff
import subprocess
# from moondream import moondream
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import random
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.tag2text import tag2text_caption
from util import *
import gradio as gr
from chatbot import *
from load_internvideo import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from simplet5 import SimpleT5
from models.grit_model import DenseCaptioning
import whisper
from ocr import ocr
import cv2
bot = ConversationBot()
image_size = 384
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((image_size, image_size)),transforms.ToTensor(),normalize])


# define model
model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size, vit='swin_b' )
model.eval()
model = model.to(device)
print("[INFO] initialize caption model success!")

model_T5 = SimpleT5()
if torch.cuda.is_available():
    model_T5.load_model(
        "t5", "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=True)
else:
    model_T5.load_model(
        "t5", "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=False)
print("[INFO] initialize summarize model success!")
# action recognition
intern_action = load_intern_action(device)
trans_action = transform_action()
topil =  T.ToPILImage()
print("[INFO] initialize InternVideo model success!")

dense_caption_model = DenseCaptioning(device)
dense_caption_model.initialize_model()
print("[INFO] initialize dense caption model success!")


speech_recognition_model_medium = whisper.load_model("medium")
speech_recognition_model_small = whisper.load_model("small")
speech_recognition_model_base = whisper.load_model("base")
speech_recognition_model_tiny = whisper.load_model("tiny")
print("[INFO] initialize speech recognition model success!")

def inference(video_path, input_tag, progress=gr.Progress()):

    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取视频的帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算视频时长
    duration = frame_count / fps if fps > 0 else 0

    video_info_message = f"Video Information:\nFrame Rate: {fps:.2f} FPS\nTotal Frames: {frame_count}\nDuration: {duration:.2f} seconds"
    
    # 释放VideoCapture对象
    cap.release()

    print(f"The duration of the video is {duration:.2f} seconds.")

    if duration<120:
        speech_recognition_model = speech_recognition_model_medium
    elif duration<360:
        speech_recognition_model = speech_recognition_model_small
    elif duration<960:
        speech_recognition_model = speech_recognition_model_base
    else:
        speech_recognition_model = speech_recognition_model_tiny 

    # speech_recognition_model = speech_recognition_model_small
    
    progress(0.1, desc="speech recognize")
    T1 = time.time()

    result = speech_recognition_model.transcribe(video_path)
    speech_recognition = result["text"]
    print(speech_recognition)

    T2 = time.time()
    print('speech recognize time:%ss' % ((T2 - T1)))

    progress(0.3, desc="OCR")
    OCR_result = ""
    OCR_result = ocr.OCR(video_path)

    print(OCR_result)

    T3 = time.time()
    print('OCR time:%ss' % ((T3 - T2)))

    # keyframe extract
    progress(0.4, desc="extract keyframe") 
    video_path = keyframes_extract_diff.extract_keyframes(video_path)
    data = loadvideo_decord_origin(video_path)
    T4 = time.time()
    print('keyframe extract time:%ss' % ((T4 - T3)))

    
    # InternVideo
    progress(0.5, desc="InternVideo")
    action_index = np.linspace(0, len(data)-1, 8).astype(int)
    tmp,tmpa = [],[]
    for i,img in enumerate(data):
        tmp.append(transform(img).to(device).unsqueeze(0))
        if i in action_index:
            tmpa.append(topil(img))
    action_tensor = trans_action(tmpa)
    TC, H, W = action_tensor.shape
    action_tensor = action_tensor.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4).to(device)
    with torch.no_grad():
        prediction = intern_action(action_tensor)
        prediction = F.softmax(prediction, dim=1).flatten()
        prediction = kinetics_classnames[str(int(prediction.argmax()))]

    T5 = time.time()
    print('InternVideo time:%ss' % ((T5 - T4)))

    # dense caption
    progress(0.6, desc="dense caption")
    dense_caption = []
    dense_index = np.arange(0, len(data)-1, 1)
    original_images = data[dense_index,:,:,::-1]
    with torch.no_grad():
        for original_image in original_images:
            dense_caption.append(dense_caption_model.run_caption_tensor(original_image))
            total_caption_num = len(dense_caption)
        
        interval = duration / total_caption_num  # 计算间隔
        dense_index = [round(i * interval) for i in range(total_caption_num)]
        dense_caption = ' '.join([f"Second {i+1} : {j}.\n" for i,j in zip(dense_index,dense_caption)])
    
    T6 = time.time()
    print('dense caption time:%ss' % ((T6 - T5)))

    # Video Caption
    progress(0.7, desc="video caption")
    image = torch.cat(tmp).to(device)   
    model.threshold = 0.68
    if input_tag == '' or input_tag == 'none' or input_tag == 'None':
        input_tag_list = None
    else:
        input_tag_list = []
        input_tag_list.append(input_tag.replace(',',' | '))
    with torch.no_grad():
        caption, tag_predict = model.generate(image,tag_input = input_tag_list,max_length = 50, return_tag_predict = True)
        progress(0.6, desc="Watching Videos")

        interval = duration / len(caption)  # 计算间隔
        dense_index = [round(i * interval) for i in range(len(caption))]

        frame_caption = ' '.join([f"Second {i+1}:{j}.\n" for i,j in zip(dense_index, caption)])
        if input_tag_list == None:
            tag_1 = set(tag_predict)
            tag_2 = ['none']
        else:
            _, tag_1 = model.generate(image,tag_input = None, max_length = 50, return_tag_predict = True)
            tag_2 = set(tag_predict)
        progress(0.8, desc="Understanding Videos")
        synth_caption = model_T5.predict('. '.join(caption))
    print(frame_caption, dense_caption, synth_caption)
    del data, action_tensor, original_image, image,tmp,tmpa
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    T7 = time.time()
    print('Video Caption time:%ss' % ((T7 - T6)))

    # frame descriptions
    progress(0.8, desc="frame descriptions") 
    result = subprocess.run(['python', 'moondream\moondream.py', str(total_caption_num)+" "+str(fps)],capture_output=True, text=True)
    frame_descriptions = result.stdout
    print(frame_descriptions)

    T8 = time.time()
    print('frame descriptions time:%ss' % ((T8 - T7)))

    return ' | '.join(tag_1),' | '.join(tag_2), frame_caption, dense_caption,synth_caption[0], gr.update(interactive = True), prediction, speech_recognition, OCR_result, frame_descriptions, video_info_message

def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])


with gr.Blocks(css="#chatbot {overflow:auto; height:500px;}") as demo:
    gr.Markdown("<h1><center>Ask Anything with GPT</center></h1>")
    gr.Markdown(
        """
        Ask-Anything is a multifunctional video question answering tool that combines the functions of Action Recognition, Visual Captioning and ChatGPT. Our solution generates dense, descriptive captions for any object and action in a video, offering a range of language styles to suit different user preferences. It supports users to have conversations in different lengths, emotions, authenticity of language.<br>  
        <p><a href='https://github.com/OpenGVLab/Ask-Anything'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_video_path = gr.inputs.Video(label="Input Video")
            input_tag = gr.Textbox(lines=1, label="User Prompt (Optional, Enter with commas)",visible=False)
          
            with gr.Row():
                with gr.Column(sclae=0.3, min_width=0):
                    caption = gr.Button("✍ Upload")
                    chat_video = gr.Button(" 🎥 Let's Chat! ", interactive=False)
                with gr.Column(scale=0.7, min_width=0):
                    loadinglabel = gr.Label(label="State")
        with gr.Column():
            openai_api_key_textbox = gr.Textbox(
                value=os.environ["OPENAI_API_KEY"],
                placeholder="Paste your OpenAI API key here to start (sk-...)",
                show_label=False,
                lines=1,
                type="password",
            )
            chatbot = gr.Chatbot(elem_id="chatbot", label="gpt")
            state = gr.State([])
            user_tag_output = gr.State("")
            image_caption_output = gr.State("")
            video_caption_output  = gr.State("")
            model_tag_output = gr.State("")
            dense_caption_output = gr.State("")
            speech_recognition = gr.State("")
            OCR_result = gr.State("")
            frame_descriptions = gr.State("")
            video_info_message = gr.State("")
            with gr.Row(visible=False) as input_raws:
                with gr.Column(scale=0.8):
                    txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
                with gr.Column(scale=0.10, min_width=0):
                    run = gr.Button("🏃‍♂️Run")
                with gr.Column(scale=0.10, min_width=0):
                    clear = gr.Button("🔄Clear️")    

    with gr.Row():
            example_videos = gr.Dataset(components=[input_video_path], samples=[['images/yoga.mp4'], ['images/making_cake.mp4'], ['images/playing_guitar.mp4']])

    example_videos.click(fn=set_example_video, inputs=example_videos, outputs=example_videos._components)
    caption.click(bot.memory.clear)
    caption.click(lambda: gr.update(interactive = False), None, chat_video)
    caption.click(lambda: [], None, chatbot)
    caption.click(lambda: [], None, state)    
    caption.click(inference,[input_video_path,input_tag],[model_tag_output, user_tag_output, image_caption_output, dense_caption_output,video_caption_output, chat_video, loadinglabel, speech_recognition, OCR_result, frame_descriptions, video_info_message])

    chat_video.click(bot.init_agent, [openai_api_key_textbox, image_caption_output, dense_caption_output, video_caption_output, model_tag_output, state, speech_recognition, OCR_result, frame_descriptions, video_info_message], [input_raws,chatbot, state, openai_api_key_textbox])

    txt.submit(bot.run_text, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)
    run.click(bot.run_text, [txt, state], [chatbot, state])
    run.click(lambda: "", None, txt)

    clear.click(bot.memory.clear)
    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)
    


demo.launch(server_name="127.0.0.1",enable_queue=True)#,share=True)
