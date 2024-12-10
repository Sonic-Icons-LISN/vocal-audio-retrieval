import re

def get_custom_metadata(info, audio):
    prompt = info["relpath"]
    prompt = prompt.replace("_", " ")
    prompt = prompt.replace("-", " ")
    prompt = prompt.replace("/", " ")
    prompt = prompt.replace(".wav", "")

    result = re.sub(r'kshmr', '', prompt, flags=re.IGNORECASE)
    return {"prompt": info["relpath"]}
