## download model
## apt install aria2
## /bin/bash
#mkdir miulab_llama2
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/resolve/main/pytorch_model-00001-of-00003.bin -d ./miulab-llama2/ -o pytorch_model-00001-of-00003.bin
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/resolve/main/pytorch_model-00002-of-00003.bin -d ./miulab-llama2/ -o pytorch_model-00002-of-00003.bin
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/resolve/main/pytorch_model-00003-of-00003.bin -d ./miulab-llama2/ -o pytorch_model-00003-of-00003.bin
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/raw/main/pytorch_model.bin.index.json -d ./miulab-llama2/ -o pytorch_model.bin.index.json
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/raw/main/special_tokens_map.json -d ./miulab-llama2/ -o special_tokens_map.json
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/resolve/main/tokenizer.model -d ./miulab-llama2/ -o tokenizer.model
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/raw/main/tokenizer_config.json -d ./miulab-llama2/ -o tokenizer_config.json
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/raw/main/config.json -d ./miulab-llama2/ -o config.json
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/raw/main/generation_config.json -d ./miulab-llama2/ -o generation_config.json
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0/raw/main/tokenizer.json -d ./miulab-llama2/ -o tokenizer.json

## pip install vllm
from vllm import LLM, SamplingParams

samp = SamplingParams(temperature=0.9)
llm = LLM(model="./miulab_llama2/", dtype="half", tensor_parallel_size=4) # tensor_parallel_size -> num of GPU
while True:
  s = input("User:")
  if s.strip() == '':
    print("bye")
    break
  response = llm.generate(s, samp)
  response_text = response[0].outputs[0].text
  print(response_text)
