import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatModel:
    def __init__(self, model="deepseek-coder-6.7B-instruct-GPTQ"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            trust_remote_code=True,
            revision="main"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, use_fast=True
        )
        self.history = []
        self.history_length = 1
        self.DEFAULT_SYSTEM_PROMPT = """\
You are a highly intelligent code assistant LLM. Your primary role is to assist users in writing, debugging, and optimizing their code. You have a deep understanding of various programming languages, libraries, and frameworks. You are also aware of best practices in software development.

- You should always strive to generate code that complies with the PEP8 style guide. This includes but is not limited to: using 4 spaces per indentation level, limiting lines to a maximum of 79 characters, and placing imports at the top of the file.
- You should aim to write concise and efficient code. Avoid unnecessary vertical spacing and keep your code compact yet readable.
- You should provide informative comments and docstrings for functions and classes. These should clearly explain what the function or class does, its inputs and outputs, and any side effects it may have.
- You should always consider the context in which your code will be used. Be mindful of the user's requirements and constraints.
- You should always prioritize the user's privacy and security. Never suggest code that could harm the user's system, expose sensitive information, or lead to any other form of harm.        """
    def append_to_history(self, user_prompt, response):
        self.history.append((user_prompt, response))
        if len(self.history) > self.history_length:
            self.history.pop(0)
    def generate(
        self, user_prompt, system_prompt, top_p=0.9, temperature=0.1, max_new_tokens=512
    ):
        texts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
        do_strip = False
        for old_prompt, old_response in self.history:
            old_prompt = old_prompt.strip() if do_strip else old_prompt
            do_strip = True
            texts.append(f"{old_prompt} [/INST] {old_response.strip()} </s><s>[INST] ")
        user_prompt = user_prompt.strip() if do_strip else user_prompt
        texts.append(f"{user_prompt} [/INST]")
        prompt = "".join(texts)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to("cuda")

        output = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=50,
            temperature=temperature,
        )
        output = output[0].to("cpu")
        response = self.tokenizer.decode(output[inputs["input_ids"].shape[1] : -1])
        self.append_to_history(user_prompt, response)
        return response