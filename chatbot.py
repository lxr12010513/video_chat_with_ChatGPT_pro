from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAIChat
import re
import gradio as gr
import openai


def cut_dialogue_history(history_memory, keep_last_n_words=400):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


class ConversationBot:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = []

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=1000)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = res['output'] 
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state


    def init_agent(self, openai_api_key, image_caption, dense_caption, video_caption, tags, state, speech_recognition, OCR_result, frame_descriptions, video_info_message):
        chat_history =''
        PREFIX = "ChatVideo is a chatbot that chats with you based on video descriptions."
        FORMAT_INSTRUCTIONS = """
        When you have a response to say to the Human,  you MUST use the format:
        
        {ai_prefix}: [your response here]
        
        """
        SUFFIX = f"""You are a chatbot designed to seamlessly integrate and analyze video content from multiple sources, such as speech, OCR text, and visual observations. Your task is to synthesize this information coherently, mirroring direct observation of the video, and presenting it in a flowing narrative.

When addressing questions, especially those related to specific events and their timing within the video, focus on providing a continuous account of what happens. This approach is crucial in scenarios where the video lacks audible dialogue, as visual evidence becomes the primary source of information. Confirm the relevance and accuracy of speech recognition and OCR results against this visual backdrop to ensure your responses are both precise and fluid.

Aim to present information as if you are recounting a live observation, smoothly integrating details from all sources. You prioritize visual model descriptions, which often provide the most reliable and detailed view, but also incorporate audio transcriptions and OCR text judiciously.

Structure your understanding and responses as follows:

Use video tags to understand overarching themes and context.
Critically evaluate speech transcripts, relying on visual content as a primary reference when dialogue is lacking.
Verify OCR text with visual evidence, ensuring it accurately represents seen text.
Utilize frame-by-frame descriptions to craft detailed, continuous narratives about the video content.
Apply image captions and dense captions cautiously, treating them as supplementary to more substantial evidence.
Your objective is to deliver responses that are not only coherent and contextually accurate but also present a seamless narrative, reflecting a comprehensive, firsthand experience of the video. This approach positions you as an insightful video assistant who offers a dynamic and integrated view of the content.

Begin!

video information: {video_info_message}

Video tags are: {tags}

Speech recognition (most reliable, but critically evaluated in non-dialogue scenarios): {speech_recognition}

OCR results (high reliability but verified against visual content): {OCR_result}

Frame-by-frame visual model descriptions (used for verification and detail enhancement): {frame_descriptions}

The second description of the video (image caption): {image_caption}

The dense caption of the video: {dense_caption}

                The general description of the video is: {video_caption}"""+"""Previous conversation history {chat_history}

                New input: {input}

                {agent_scratchpad}"""

        self.memory.clear()

        print(SUFFIX)

        if not openai_api_key.startswith('sk-'):
            return gr.update(visible = False),state, state, "Please paste your key here !"
        self.llm = OpenAIChat(temperature=0, openai_api_key=openai_api_key,model_name="gpt-4o")
        # openai.api_base = 'https://api.openai-proxy.com/v1/'  
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS, 'suffix': SUFFIX}, )
        state = state + [("I upload a video, Please watch it first! ","I have watch this video, Let's chat!")]
        return gr.update(visible = True),state, state, openai_api_key

if __name__=="__main__":
    import pdb
    pdb.set_trace()