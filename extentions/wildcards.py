import os
import sys
import re
import json
import math
import gradio as gr


#from modules.util import get_files_from_folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from args_manager import args

wildcards_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../wildcards'))
wildcards_max_bfs_depth = 64
wildcards = {}
wildcards_list = {}
wildcards_translation = {}
wildcards_template = {}
wildcards_weight_range = {}

array_regex = re.compile(r'\[([\w\(\)\.\s,;:-]+)\]')
array_regex1 = re.compile(r'\[([\w\(\)\s,;.:\"-]+)\]')
tag_regex0 = re.compile(r'([\s\w\(\);-]+)')
tag_regex1 = re.compile(r'([\s\w\(\),-]+)')
tag_regex2 = re.compile(r'__([\w-]+)__')
tag_regex3 = re.compile(r'__([\w-]+)__:([\d]+)')
tag_regex4 = re.compile(r'__([\w-]+)__:([RLrl]){1}([\d]*)')
tag_regex5 = re.compile(r'__([\w-]+)__:([RLrl]){1}([\d]*):([\d]+)')
tag_regex6 = re.compile(r'__([\w-]+)__:([\d]+):([\d]+)')

wildcard_regex = re.compile(r'-([\w-]+)-')
folder_variation = {}
def get_files_from_folder(folder_path, extensions=None, name_filter=None, variation=False):
    global folder_variation
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []
    for root, dirs, files in os.walk(folder_path, topdown=False):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in sorted(files, key=lambda s: s.casefold()):
            _, file_extension = os.path.splitext(filename)
            if (extensions is None or file_extension.lower() in extensions) and (name_filter is None or name_filter in _):
                path = os.path.join(relative_path, filename)
                if variation:
                    mtime = int(os.path.getmtime(os.path.join(root, filename)))
                    if folder_path not in folder_variation or path not in folder_variation[folder_path] or mtime > folder_variation[folder_path][path]:
                        if folder_path not in folder_variation:    
                            folder_variation.update({folder_path: {path: mtime}})
                        else:
                            folder_variation[folder_path].update({path: mtime})
                        filenames.append(path)
                else:
                    filenames.append(path)


    return filenames
def set_wildcard_path_list(name, list_value):
    global wildcards_list
    if name in wildcards_list.keys():
        if list_value not in wildcards_list[name]:
            wildcards_list[name].append(list_value)
    else:
        wildcards_list.update({name: [list_value]})

def get_wildcards_samples(path="root"):
    global wildcards_path, wildcards, wildcards_list, wildcards_translation, wildcards_template, wildcards_weight_range, wildcard_regex

    wildcards_list_all = sorted([f[:-4] for f in get_files_from_folder(wildcards_path, ['.txt'], None, variation=True)])
    wildcards_list_all = [x for x in wildcards_list_all if not (x.startswith('_') or x.endswith('_'))]
    #print(f'wildcards_list:{wildcards_list_all}')
    for wildcard in wildcards_list_all:
        words = open(os.path.join(wildcards_path, f'{wildcard}.txt'), encoding='utf-8').read().splitlines()
        words = [x.split('?')[0] for x in words if x != '' and not wildcard_regex.findall(x)]
        #words = [x.split(';')[0] for x in words]

        templates = [x for x in words if '|' in x]  #  word|template|weight_range
        for line in templates:
            parts = line.split("|")
            word = parts[0]
            template = parts[1]
            weight_range = ''
            if len(parts)>2:
                weight_range = parts[2]
            if word is None or word == '':
                wildcards_template.update({wildcard: template})
                if len(weight_range.strip())>0:
                    wildcards_weight_range.update({wildcard: weight_range})
            else:
                wildcards_template({f'{wildcard}/{word}': template})
                if len(weight_range.strip())>0:
                    wildcards_weight_range.update({f'{wildcard}/{word}': weight_range})
        words = [x.split("|")[0] for x in words]
        wildcards.update({wildcard: words})
        wildcard_path = wildcard.split("/")
        if len(wildcard_path)==1:
            set_wildcard_path_list("root", wildcard_path[0])
        elif len(wildcard_path)==2:
            set_wildcard_path_list(wildcard_path[0], wildcard_path[1])
            #set_wildcard_path_list("root", wildcard_path[0])
        elif len(wildcard_path)==3:
            set_wildcard_path_list(f'{wildcard_path[0]}/{wildcard_path[1]}', wildcard_path[2])
            set_wildcard_path_list(wildcard_path[0], wildcard_path[1])
            #set_wildcard_path_list("root", wildcard_path[0])
        else:
            print(f'[Wildcards] The level of wildcards is too depth: {wildcards_path}.')
    #print(f'wildcards_list:{wildcards_list}')
    if wildcards_list_all:
        print(f'[Wildcards] Refresh and Load {len(wildcards_list_all)}/{len(wildcards.keys())} wildcards: {", ".join(wildcards_list_all)}.')
    

    return [[x] for x in wildcards_list[path]]

def get_words_of_wildcard_samples(wildcard="root"):
    global wildcards, wildcards_list

    if wildcard == "root":
        return [[x] for x in wildcards[wildcards_list[wildcard][0]]]
    words = [[x] for x in wildcards[wildcard]]
    return words

def get_words_with_wildcard(wildcard, rng, method='R', number=1, start_at=1):
    global wildcards

    words = wildcards[wildcard]
    words_result = []
    number0 = number
    if method=='L' or method=='l':
        if number == 0:
            words_result = words
        else:
            if number < 0:
                number = 1
            start = start_at - 1
            if number > len(words):
                number = len(words)
            if (start + number)>len(words):
                words_result = words[start:] + words[:start + number - len(words)]
            else:
                words_result = words[start:start + number]
    else:
        if number < 1:
            number = 1
        if number > len(words):
            number = len(words)
        nums = 1 if start_at<=1 else start_at
        for i in range(number):
            words_each = rng.sample(words, nums)
            words_result.append(words_each[0] if nums==1 else f'({" ".join(words_each)})')
    words_result = [replace_wildcard(txt, rng) for txt in words_result]
    print(f'[Wildcards] Get words from wildcard:__{wildcard}__, method:{method}, number:{number}, start_at:{start_at}, result:{words_result}')
    return words_result


def compile_arrays(text, rng):
    global wildcards, wildcards_max_bfs_depth, array_regex, tag_regex1, tag_regex2, tag_regex3, tag_regex4, tag_regex5

    _ = get_wildcards_samples()
    tag_arrays = array_regex.findall(text)
    arrays = []
    mult = 1
    seed_fixed = True
    if len(tag_arrays)>0:
        for tag in tag_arrays:
            colon_counter = tag.count(':')
            wildcard = ''
            number = 1
            method = 'R'
            start_at = 1
            if colon_counter == 2:
                parts = tag_regex5.findall(tag)
                if parts:
                    parts = list(parts[0])
                    wildcard = parts[0]
                    method = parts[1]
                    if parts[2]:
                        number = int(parts[2])
                    start_at = int(parts[3])
                else:
                    parts = tag_regex6.findall(tag)
                    if parts:
                        parts = list(parts[0])
                        wildcard = parts[0]
                        number = int(parts[1])
                        start_at = int(parts[2])
            elif colon_counter == 1:
                parts = tag_regex3.findall(tag)
                if parts:
                    parts = list(parts[0])
                    wildcard = parts[0]
                    number = int(parts[1])
                else:
                    parts = tag_regex4.findall(tag)
                    if parts:
                        parts = list(parts[0])
                        wildcard = parts[0]
                        method = parts[1]
                        if parts[2]:
                            number = int(parts[2])
            elif colon_counter == 0:
                parts = tag_regex1.findall(tag)
                if parts:
                    words = parts[0].split(',')
                    words = [x.strip() for x in words]
                    text = text.replace(tag, ','.join(words))
                    arrays.append(words)
                    mult *= len(words)
                    continue
                else:
                    parts = tag_regex0.findall(tag)
                    if parts:
                        words = parts[0].split(';')
                        words = [x.strip() for x in words]
                        text = text.replace(tag, ';'.join(words))
                        arrays.append(words)
                        mult *= len(words)
                        seed_fixed = False
                        continue
            words = get_words_with_wildcard(wildcard, rng, method, number, start_at)
            delimiter = ',' if method.isupper() else ';'
            text = text.replace(tag, delimiter.join(words), 1)
            arrays.append(words)
            mult *= len(words)
            if delimiter == ';':
                seed_fixed = False
    else:
        mult = 0
    

    print(f'[Wildcards] Copmile text in prompt to arrays: {text} -> arrays:{arrays}, mult:{mult}')
    return text, arrays, mult, seed_fixed

def replace_wildcard(text, rng):
    global wildcards_max_bfs_depth, tag_regex2, wildcards
    parts = tag_regex2.findall(text)
    i = 1
    while parts:
        for wildcard in parts:
            text = text.replace(f'__{wildcard}__', rng.choice(wildcards[wildcard]), 1)
        parts = tag_regex2.findall(text)
        i += 1
        if i > wildcards_max_bfs_depth:
            break
    return text


def get_words(arrays, totalMult, index):
    if(len(arrays) == 1):
        word = arrays[0][index]
        if word[0] == '(' and word[-1] == ')':
            word = word[1:-1]
        return [word]
    else:
        words = arrays[0]
        word = words[index % len(words)]
        if word[0] == '(' and word[-1] == ')':
            word = word[1:-1]
        index -= index % len(words)
        index /= len(words)
        index = math.floor(index)
        return [word] + get_words(arrays[1:], math.floor(totalMult/len(words)), index)


def apply_arrays(text, index, arrays, mult):
    if len(arrays) == 0:
        return text
    
    tags = array_regex1.findall(text)
    
    index %= mult
    chosen_words = get_words(arrays, mult, index)

    i = 0
    for arr in arrays:
        if i<len(tags) and i<len(chosen_words):
            if not tag_regex2.findall(chosen_words[i]):
                text = text.replace(f'[{tags[i]}]', chosen_words[i], 1)
            else:
                text = text.replace(f'[{tags[i]}]', tags[i], 1)
        i = i+1

    return text


def apply_wildcards(wildcard_text, rng, directory=wildcards_path):
    global tag_regex2, wildcards

    for _ in range(wildcards_max_bfs_depth):
        placeholders = tag_regex2.findall(wildcard_text)
        if len(placeholders) == 0:
            return wildcard_text

        print(f'[Wildcards] processing: {wildcard_text}')
        for placeholder in placeholders:
            try:
                words = wildcards[placeholder]
                assert len(words) > 0
                wildcard_text = wildcard_text.replace(f'__{placeholder}__', rng.choice(words), 1)
            except:
                print(f'[Wildcards] Warning: {placeholder}.txt missing or empty. '
                      f'Using "{placeholder}" as a normal word.')
                wildcard_text = wildcard_text.replace(f'__{placeholder}__', placeholder)
            print(f'[Wildcards] {wildcard_text}')

    print(f'[Wildcards] BFS stack overflow. Current text: {wildcard_text}')
    return wildcard_text


def add_wildcards_and_array_to_prompt(wildcard, prompt, state_params):
    global wildcards, wildcards_list

    wildcard = wildcard[0].split('|')[0]
    state_params.update({"wildcard_in_wildcards": wildcard})
    if len(prompt)==0:
        prompt=' '
    if prompt[-1]=='[':
        state_params["array_wildcards_mode"] = '['
        prompt = prompt[:-1]
    elif prompt[-1]=='(':
        state_params["array_wildcards_mode"] = '_'
#        prompt = prompt[:-1]
    elif prompt[-1]=='_':
        state_params["array_wildcards_mode"] = '_'
        if len(prompt)==1 or len(prompt)>2 and prompt[-2]!='_':
            prompt = prompt[:-1]
    else:
        state_params["array_wildcards_mode"] = '_'
        prompt += ' '

    
    if state_params["array_wildcards_mode"] == '[':
#        new_tag = f'[__{wildcard}__]'
        new_tag = f'['
    else:
        new_tag = f'__{wildcard}__'
        
    prompt = f'{prompt.strip()}{new_tag}'
    return gr.update(value=prompt), gr.Dataset.update(label=f'{wildcard}:', samples=get_words_of_wildcard_samples(wildcard)), gr.update(open=True)

def add_word_to_prompt(wildcard, index, prompt):
    global wildcards, wildcards_list

    wildcard = wildcard[0].split('|')[0]
    words = wildcards[wildcard]
    word = words[index]
    prompt = f'{prompt.strip()} {word}'
    return gr.update(value=prompt)

