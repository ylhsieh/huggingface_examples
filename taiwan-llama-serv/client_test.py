# coding=utf-8
import json
import time

import urllib.request
from typing import Iterable, List
import sys
import requests

system_prompt = \
"""A chat between a curious user and an artificial intelligence assistant. 
The assistant writes a short summary of the document.
DOCUMENT:\n
多發性硬化症（英語：Multiple sclerosis，縮寫：MS）是一種脫髓鞘性神經病變，患者腦或脊髓中的神
經細胞表面的絕緣物質（即髓鞘）受到破壞，神經系統的訊息傳遞受損，導致一系列可能發生的症狀，影響患
者的活動、心智、甚至精神狀態。這些症狀可能包括複視、單側視力受損、肌肉無力、感覺遲鈍，或協調障礙。
多發性硬化症的病情多變，患者的症狀可能反覆發作，也可能持續加劇。在每次發作之間，症狀有可能完全消失，
但永久性的神經損傷仍然存在，這在病情嚴重的患者特別明顯。雖然具體的成因不明，但多發性硬化症的機制
可能為髓鞘受到免疫系統破壞或生成髓鞘的細胞發生問題，可能的原因包括遺傳與環境因素，
例如受病毒感染的刺激而引發自體免疫反應。多發性硬化症的診斷需藉助臨床表現和相關的影像學證據支持。
格林-巴厘綜合徵(Guillain-Barre Syndrome,GBS)，是一種急性炎症性脫髓鞘性多發性神經病變
(acute inflammatory demyelinating polyradiculoneuropathies, AIDP)，乃是由自身
免疫系統傷害週邊神經所引起的疾病。GBS發生率大約為每年每10萬人口約1-2人會得病，算是罕見。
"""

def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8", "ignore"))
            output = data["text"]
            yield output

def gen_prompt(message: str, chat_history: list[tuple[str, str]] = []) -> str:
    texts = [f'{system_prompt}\n']
    for user_input, response in chat_history:
        texts.append(f'USER: {user_input.strip()}\nASSISTANT: {response}\n')
    texts.append(f'USER: {message.strip()}\nASSISTANT: ')
    return ''.join(texts)

def call_api(input_text):
    api_url = 'http://127.0.0.1:8090/generate'
    header = {'Content-Type': 'application/json'}

    prompt = gen_prompt(input_text.strip())
    prompt_len = len(prompt)
    data = {
          "prompt": prompt,
          "stream" : True,
          "n" : 1,
          "best_of": 1, 
          "presence_penalty": 0.0, 
          "frequency_penalty": 0.1, 
          "temperature": 0.5, # A higher temperature: more diverse and creative but might also increase its likelihood of straying from the context.
          "top_p" : 0.9, 
          "top_k": 40, 
          "use_beam_search": False, 
          "stop": [], 
          "ignore_eos" :False, 
          "max_tokens": 4096, 
          "logprobs": None
    }
    request = urllib.request.Request(
        url=api_url,
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )

    result = None
    try:
        
        ## non streaming
        # response = urllib.request.urlopen(request, timeout=300)
        # res = response.read().decode('utf-8')
        # result = json.loads(res)
        ## print(json.dumps(data, ensure_ascii=False, indent=2))
        ## print(json.dumps(result, ensure_ascii=False, indent=2))
        ## print(result['response'][0])
        # for r in result['responses']:
        #     print(r)


        # streaming
        response = requests.post(api_url, headers=header, json=data, stream=True)
        num_printed_chars = 0
        for h in get_streaming_response(response):
            for i, line in enumerate(h):
                # handle Chinese decode error
                if line[-1] == "\uFFFD": continue
                # print(f"Beam candidate {i}: {line_crop!r}", flush=True)
                print(f"{line[num_printed_chars:]}", flush=True, end='')
                num_printed_chars = len(line)
        print("\n")

    except Exception as e:
        print(e)        
        
    return result

if __name__ == "__main__":

    call_api("Summarize")
