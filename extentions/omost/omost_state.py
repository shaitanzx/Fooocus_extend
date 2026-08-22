"""
Модуль состояния Omost — общие глобальные переменные для main.py и async_worker.py.
Позволяет async_worker обнулять состояние LLM через коллбэк.
"""

# Глобальные переменные состояния LLM
llm_patcher = None
llm_model = None
llm_tokenizer = None
llm_name = None
omost_seed = None


def reset_llm_state():
    """
    Обнуляет все глобальные переменные состояния LLM.
    Вызывается из async_worker при удалении LLM через remove_llm_from_memory().
    """
    global llm_patcher, llm_model, llm_tokenizer, llm_name
    
    llm_patcher = None
    llm_model = None
    llm_tokenizer = None
    llm_name = None
    
    print(f"[OmostState] ✓ Глобальные переменные LLM обнулены")


def get_llm_info():
    """Возвращает информацию о текущем состоянии LLM."""
    return {
        'patcher': llm_patcher,
        'model': llm_model,
        'tokenizer': llm_tokenizer,
        'name': llm_name,
    }