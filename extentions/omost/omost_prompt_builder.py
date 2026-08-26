import ast
import json
import re

MODE_CONFIG = {
    'normal':     {'include_details': True,  'dedup_substrings': False},
    'aggressive': {'include_details': True,  'dedup_substrings': True},
    'short':      {'include_details': False, 'dedup_substrings': False},
}

DEFAULT_MODE = 'normal'

def extract_code_from_response(llm_response):
    if not llm_response:
        return ""

    text = llm_response.strip()

    code_match = re.search(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    if 'canvas = Canvas()' not in text:
        start = text.find('canvas')
        if start >= 0:
            return text[start:].strip()

    return text


def parse_canvas_code(llm_response):
    if not llm_response:
        print('[CanvasParser] Empty response')
        return None

    code = extract_code_from_response(llm_response)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f'[CanvasParser] Syntax error: {e}')
        return None

    result = {'global': None, 'regions': []}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue

        method_name = node.func.attr

        kwargs = {}
        for kw in node.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                continue

        if method_name == 'set_global_description':
            result['global'] = kwargs
        elif method_name == 'add_local_description':
            result['regions'].append(kwargs)

    print(f'[CanvasParser] Parsed: global={"yes" if result["global"] else "no"}, '
          f'regions={len(result["regions"])}')
    return result


def build_layout(parsed):

    if not parsed or not parsed.get('global'):
        print('[LayoutBuilder] No global description')
        return {}

    g = parsed['global']

    layout = {
        'global': {'prefixes': [], 'suffixes': []},
        'regions': []
    }

    if g.get('description'):
        layout['global']['prefixes'].append(g['description'])

    if g.get('detailed_descriptions'):
        layout['global']['suffixes'].extend(g['detailed_descriptions'])
    if g.get('tags'):
        layout['global']['suffixes'].append(g['tags'])
    if g.get('atmosphere'):
        layout['global']['suffixes'].append(g['atmosphere'])
    if g.get('style'):
        layout['global']['suffixes'].append(g['style'])
    if g.get('quality_meta'):
        layout['global']['suffixes'].append(g['quality_meta'])

    for r in parsed.get('regions', []):
        region = {
            'distance_to_viewer': r.get('distance_to_viewer', 0.0),
            'location': r.get('location', ''),
            'area': r.get('area', ''),
            'prefixes': [],
            'suffixes': []
        }

        if r.get('description'):
            region['prefixes'].append(r['description'])
        if r.get('detailed_descriptions'):
            region['suffixes'].extend(r['detailed_descriptions'])
        if r.get('tags'):
            region['suffixes'].append(r['tags'])
        if r.get('atmosphere'):
            region['suffixes'].append(r['atmosphere'])
        if r.get('style'):
            region['suffixes'].append(r['style'])
        if r.get('quality_meta'):
            region['suffixes'].append(r['quality_meta'])

        layout['regions'].append(region)

    print(f'[LayoutBuilder] Built layout: regions={len(layout["regions"])}')
    return layout

def deduplicate_phrases(phrases, remove_substrings=False):

    seen = set()
    out = []
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        clean = phrase.strip().rstrip(".").strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)

    if not remove_substrings:
        return out

    final = []
    for i, phrase in enumerate(out):
        phrase_lower = phrase.lower()
        is_substring = False
        for j, other in enumerate(out):
            if i == j:
                continue
            other_lower = other.lower()
            if len(phrase_lower) < len(other_lower) and phrase_lower in other_lower:
                is_substring = True
                break
        if not is_substring:
            final.append(phrase)

    removed = len(out) - len(final)
    if removed:
        print(f'[Dedup] Removed {removed} substring phrases')

    return final

def flatten_to_prompt(layout, mode=DEFAULT_MODE, separator=", "):

    if not isinstance(layout, dict) or "global" not in layout:
        print('[Flatten] Invalid layout: missing "global"')
        return ""

    config = MODE_CONFIG.get(mode, MODE_CONFIG[DEFAULT_MODE])
    include_details = config['include_details']
    dedup_substrings = config['dedup_substrings']

    phrases = []

    # Global
    global_block = layout.get("global", {})
    phrases.extend(global_block.get("prefixes", []))
    if include_details:
        phrases.extend(global_block.get("suffixes", []))

    # Regions 
    regions = layout.get("regions", [])
    regions = sorted(regions, key=lambda r: r.get("distance_to_viewer", 0.0))

    for region in regions:
        phrases.extend(region.get("prefixes", []))
        if include_details:
            phrases.extend(region.get("suffixes", []))

    print(f'[Flatten] Collected {len(phrases)} phrases (mode={mode})')

    out = deduplicate_phrases(phrases, remove_substrings=dedup_substrings)

    print(f'[Flatten] After dedup: {len(out)} phrases')
    return separator.join(out)


def process_canvas(mode,llm_response, separator=", ", verbose=True):

    if mode not in MODE_CONFIG:
        print(f'[Chain] WARNING: Unknown mode "{mode}", using "{DEFAULT_MODE}"')
        mode = DEFAULT_MODE

    if verbose:
        print('\n' + '=' * 70)
        print(f'[Chain] START: mode={mode}')
        print('=' * 70)

    parsed = parse_canvas_code(llm_response)
    if not parsed:
        print('[Chain] FAILED: parse error')
        return ""

    layout = build_layout(parsed)
    if not layout:
        print('[Chain] FAILED: layout error')
        return ""

    prompt = flatten_to_prompt(layout, mode=mode, separator=separator)

    if verbose:
        print(f'\n[Chain] RESULT: {len(prompt)} chars, '
              f'{len(prompt.split()) if prompt else 0} words')
        print('=' * 70)
        print('[Chain] END')
        print('=' * 70 + '\n')

    return prompt


def debug_canvas(llm_response):
    parsed = parse_canvas_code(llm_response)
    if not parsed:
        return None
    layout = build_layout(parsed)
    print('\n' + json.dumps(layout, indent=2, ensure_ascii=False))
    return layout