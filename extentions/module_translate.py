## --- Title: Prompt translte
## --- Description: Extension for Fooocus, translated prompt to other languages
## --- Author: AlekPet (https://github.com/AlekPet/Fooocus_Extensions_AlekPet)


from deep_translator import GoogleTranslator

def translate(prompt, prompt_neg='', srcTrans="auto", toTrans="en"):
    if not srcTrans:
        srcTrans = 'auto'
        
    if not toTrans:
        toTrans = 'en'

    tranlate_text_prompt = ''
    if prompt and prompt.strip()!="":
        tranlate_text_prompt = GoogleTranslator(source=srcTrans, target=toTrans).translate(prompt) 

    tranlate_text_prompt_neg = ''
    if prompt_neg and prompt_neg.strip()!="":
        tranlate_text_prompt_neg = GoogleTranslator(source=srcTrans, target=toTrans).translate(prompt_neg) 

    return [tranlate_text_prompt, tranlate_text_prompt_neg]

