from ollama import chat
from ollama import ChatResponse
from collections import defaultdict
import re


# experiments model='llama3.2', phi4
def ask_ollama(request):
    response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': request,
        },
    ])

    return response.message.content


def get_request(sentence, key):
    request = (
        f"Considering the given sentence, identify the exact word or phrase that is most closely related to the keyphrase. "
        f"Reply with exactly the relevant word or phrase as it appears in the sentence. Do not rephrase, add explanations, or infer information from general knowledge. "
        f"If no exact match is found, reply with '<not found>'. \n"

        f"Example 1. "
        f"Sentence: ____________________ airport is located in _______ bengo which is part of _______________. the runway length of ____________________ airport is 3800. "
        f"Keyphrase: angola "
        f"You should reply: <not found> \n"

        f"Example 2. "
        f"Sentence: __________________ played with the fc torpedo moscow part of the soviet union national football "
        f"club whose chairman is __________________. that club was part of the 2014-15 russian premier league. "
        f"Keyphrase: 2014-15 russian "
        f"You should reply: 2014-15 russian \n"

        f"Example 3. "
        f"Sentence: synthpop has origins in disco and house and is a form of pop. it is the style of artist alex day. "
        f"Keyphrase: pop music "
        f"You should reply: pop \n"

        f"Find the most related phrase in the following sentence. "
        f"Sentence: '{sentence}' "
        f"Keyphrase: {key}")

    return request


def find_phrase(sentence, key, times=10):
    response_dict = defaultdict(int)
    for i in range(times):
        response_split = ask_ollama(get_request(sentence, key)).split()
        response_short = " ".join(response_split[:2]).lower()
        response_clean = re.sub(r"[<>']", "", response_short)
        response_dict[response_clean] += 1
    result = max(response_dict, key=response_dict.get)
    return result
