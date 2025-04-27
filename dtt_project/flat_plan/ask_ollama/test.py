from ollama import chat
from ollama import ChatResponse
from collections import defaultdict
import re


# experiments model='llama3.2', phi4
def ask_ollama(request):
    response: ChatResponse = chat(model='phi4', messages=[
        {
            'role': 'user',
            'content': request,
        },
    ])
    print(response.message.content)


def get_request(sentence, key):
    request = (
        f"Considering the given sentence, identify the exact word or phrase that is most closely related to the keyphrase."
        f"Reply with exactly the relevant word or phrase as it appears in the sentence. Do not rephrase, add explanations, or infer information from general knowledge."
        f"If no exact match is found, reply with '<not found>'."

        f"Example 1."
        f"Sentence: ____________________ airport is located in _______ bengo which is part of _______________. the runway length of ____________________ airport is 3800."
        f"Keyphrase: angola"
        f"You should reply: <not found>"

        f"Example 2."
        f"Sentence: __________________ played with the fc torpedo moscow part of the soviet union national football "
        f"club whose chairman is __________________. that club was part of the 2014-15 russian premier league."
        f"Keyphrase: 2014-15 russian"
        f"You should reply: 2014-15 russian"

        f"Example 3."
        f"Sentence: synthpop has origins in disco and house and is a form of pop. it is the style of artist alex day."
        f"Keyphrase: pop music"
        f"You should reply: pop"

        f"Find the most related phrase in the following sentence."
        f"Sentence: '{sentence}'"
        f"Keyphrase: {key}")

    return request


def find_phrase(sentence, key, times=10):
    response_dict = defaultdict(int)
    for i in range(times):
        response_split = ask_ollama(get_request(sentence, key)).split()
        response_short = " ".join(response_split[:2]).lower()
        response_clean = re.sub(r"[<>]", "", response_short)
        response_dict[response_clean] += 1
    result = max(response_dict, key=response_dict.get)
    return result


key1 = "united states"
sentence1 = "______________ airport is located in texas u.s. whose citizens are tejano. spanish is spoken there."

sentence2 = "_________________ who once represented the italian under 19 football team later played for ac milan managed by _________________ in serie a."
key2 = "italy national"

sentence3 = "adam mcquaid was born in october 12 1986 in p.e.i. he is 1.9558 m. high."
key3 = "prince edward"

key4 = "january 2009"
sentence4 = "____________ st. was started in jan. 2009 and completed in april 2014. it has 34 floors and is 62145.3 square meters."

key5 = "solo singer"
sentence5 = "rhythm and blues singer _____________ is a solo performer. r and b came from the blues and later gave rise to funk."

sentence6 = "the city of lahore is served by ____________ international airport which is governed by the ______________ aviation authority and has a runway length of 3310.0."
key6 = "pakistan"

sentence7 = "____________ is associated with the musical artist ___________________ and bobina. his musical genre is trance which has its origins in pop music."
key7 = "trance music"

sentence8 = "__________ was a _____________ national born in dallas. he died in _________."
key8 = "deceased"

sentence9 = "____________________ airport is located in _______ bengo which is part of _______________. the runway length of ____________________ airport is 3800."
key9 = "angola"

sentences = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, sentence7, sentence8, sentence9]
keys = [key1, key2, key3, key4, key5, key6, key7, key8, key9]

for sent, key in zip(sentences, keys):
    print(find_phrase(sent, key))
