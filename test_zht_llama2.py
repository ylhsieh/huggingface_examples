from vllm import LLM, SamplingParams

samp = SamplingParams(temperature=0.9)
llm = LLM(model="../miulab_llama2/", dtype="half", tensor_parallel_size=4) # tensor_parallel_size -> num of GPU
while True:
  s = input("User:")
  if s.strip() == '':
    print("bye")
    break
  response = llm.generate(s, samp)
  response_text = response[0].outputs[0].text
  print(response_text)
