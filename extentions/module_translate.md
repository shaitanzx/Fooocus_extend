## What Prompt Translate is For
([Читать на русском языке](#для-чего-нужен-prompt-translate))

The extension translates the positive prompt and negative prompt using Google Translate. Translation can be done manually via a button, or you can instruct the extension to translate the text automatically during task processing. By default, the source language is detected automatically, and the translation is performed into English.

## Quick Start: Automatically Translate Text During Generation

1. Open the **`Prompt Translate`** tab.
2. Set **`From`** to `auto` if you do not want to specify the source text language manually.
3. Leave **`To`** as `en` to translate the prompt into English.
4. Enable **`Enabled`**.
5. Enter the image description in the standard positive prompt field. Fill in the negative prompt if necessary.
6. Click the standard **Generate** button.

The translation is used during task processing, before the generator parses the prompt. It does not replace the original text in the main interface field. If you need a translation that you can check or edit before generation, use the **`Translate`** button.

## Panel Elements and Their Settings

### `Enabled`

**Default:** disabled.

Enables automatic translation during prompt processing. When the checkbox is disabled, the mere fact of selecting languages does not trigger automatic translation. The manual **`Translate`** button works independently of this checkbox.

When enabled, it translates both the positive and negative prompts, provided the respective fields are not empty. The translator is called internally during task processing, so the main prompt text in the interface remains in its original language.

### `Translate`

This button triggers the translation immediately, without waiting for generation and without requiring **`Enabled`** to be turned on. It translates the positive and negative prompts into the selected languages, then writes the result back into the main prompt fields. The text is also passed to the translation preview fields.

If the source field is empty or contains only spaces, the function returns an empty string for it. Therefore, after clicking the button, an empty prompt or negative prompt will remain empty.

### `From`

**Default:** `auto`.

Select the language in which the source prompt is written. The `auto` value asks Google Translate to detect the language automatically. You can also select a specific language from the available list. The list is populated at interface startup with the language codes supported by the current version of `GoogleTranslator`.

If the source text language is known, selecting a specific language can help with short or mixed phrases. If you are unsure, leave it as `auto`.

### `To`

**Default:** `en`.

Select the target language from the same list of supported languages. The initial setting is aimed at English, which is frequently used in text descriptions for generation. Change it if you need to translate into another language.

### `🔃` Button

Swaps the selected source and target languages if `From` is set explicitly and the languages differ. If `From` is set to `auto` or the languages match, the button reverts to the default pair: `From = auto`, `To = en`.

### `See translated prompts after click Generate`

**Default:** disabled.

Shows two preview fields: `Prompt translate` and `Negative Translate`. If the checkbox is enabled, after the standard **Generate** button completes, the extension translates the text from the main fields once again and displays the result in these fields. The preview only displays the translation: it does not replace the entered prompt by itself.

This preview is tied to the standard **Generate** button. It is not connected to the separate launch buttons for Prompt Batch or Image Batch. However, the automatic translation itself can still work in batch tasks if **`Enabled`** is turned on.

## Manual Translation Before Generation

1. Leave **`Enabled`** disabled if you do not want the text to be translated again during the generation process.
2. Select a language in **`From`** or leave it as `auto`.
3. Select the target language in **`To`**.
4. Fill in the positive and, if needed, the negative prompt.
5. Click **`Translate`**.
6. Review the translated text in the main fields. Edit it if necessary and click **Generate**.

This method is convenient when you need to check the translation before sending the text to the generator. Note: manual translation overwrites the contents of the main prompt fields, so the original text will not remain in them automatically.

## Modes Where Translation Works

### Standard Generation

With **`Enabled`** turned on, the positive and negative prompts are translated in the handler of each task before further preparation for generation. When manually clicking **`Translate`**, the text changes immediately, even if **`Enabled`** is disabled.

### Prompt Batch

Prompt Batch runs multiple text tasks sequentially. For each task undergoing standard prompt processing, an enabled **`Enabled`** applies the translation with the selected `From` and `To`. If preview is used, note that it is updated only by the standard **Generate** button chain, not by the Prompt Batch launch button.

## What the Result Is

Prompt Translate does not create a separate translation file. With manual translation, the result is placed in the main prompt fields, as well as in the preview fields. With automatic translation, the text is used internally within the task; the image is saved in the standard Fooocus Extend manner.

The code explicitly uses `GoogleTranslator`, not a local translation model. This means the translation depends on the availability of the external Google Translate service via `deep-translator` and a network connection. Do not send confidential texts through this mode if you do not want to pass them to an external translator.

## If Translation Fails

- **After Generate, the prompt remains in the original language.** Check if **`Enabled`** is turned on. Selecting `From` and `To` does not enable automatic translation by itself.
- **The `Translate` button replaced my words, and I need the original back.** The manual button writes the translation directly into the main fields. Restore the original text manually or paste it again from a saved copy. To check without replacing the original, enable **`See translated prompts after click Generate`** and use the standard Generate button.
- **The translation does not appear or generation ends with an error.** The implementation calls the translator directly and does not contain its own error handling. Check your internet access and the availability of Google Translate; for short text with an ambiguous language, try selecting `From` manually, and `en` as the target.

## Для чего нужен Prompt Translate
([Read in English](#what-prompt-translate-is-for))

Расширение переводит положительный prompt и отрицательный prompt с помощью переводчика Google. Перевод можно сделать вручную кнопкой или поручить расширению автоматически переводить текст при обработке задания. По умолчанию исходный язык определяется автоматически, а перевод выполняется на английский.

## Быстрый старт: автоматически переводить текст при генерации

1. Откройте вкладку **`Prompt Translate`**.
2. Установите **`From`** в `auto`, если не хотите задавать язык исходного текста вручную.
3. Оставьте **`To`** равным `en`, чтобы перевести prompt на английский.
4. Включите **`Enabled`**.
5. Введите описание изображения в обычное поле положительного prompt. При необходимости заполните отрицательный prompt.
6. Нажмите обычную кнопку **Generate**.

Перевод используется во время обработки задания, до разбора prompt генератором. Он не заменяет исходный текст в основном поле интерфейса. Если нужен перевод, который можно проверить или отредактировать до генерации, используйте кнопку **`Translate`**.

## Элементы панели и их настройки

### `Enabled`

**По умолчанию:** выключено.

Включает автоматический перевод при обработке prompt. Когда флажок выключен, сам факт выбора языков не запускает автоматический перевод. Ручная кнопка **`Translate`** работает независимо от этого флажка.

Включённый режим переводит как положительный prompt, так и отрицательный, если соответствующие поля не пусты. Переводчик вызывается внутри обработки задания, поэтому основной текст prompt в интерфейсе остаётся исходным.

### `Translate`

Кнопка запускает перевод немедленно, не дожидаясь генерации и не требуя включать **`Enabled`**. Она переводит положительный и отрицательный prompt по выбранным языкам, затем записывает результат обратно в основные поля prompt. Текст также передаётся в поля предпросмотра перевода.

Если исходное поле пустое или содержит только пробелы, функция для него возвращает пустую строку. Поэтому после нажатия кнопки пустой prompt или negative prompt останется пустым.

### `From`

**По умолчанию:** `auto`.

Выберите язык, на котором написан исходный prompt. Значение `auto` просит Google Translate определить язык автоматически. Также можно выбрать конкретный язык из доступного списка. Список заполняется при запуске интерфейса кодами языков, поддерживаемых текущей версией `GoogleTranslator`

Если язык исходного текста известен, выбор конкретного языка может помочь при коротких или смешанных фразах. Если сомневаетесь, оставьте `auto`.

### `To`

**По умолчанию:** `en`.

Выберите язык результата из того же списка поддерживаемых языков. Изначальная настройка направлена на английский, который часто используют в текстовых описаниях для генерации. Смените её, если нужно перевести на другой язык.

### Кнопка `🔃`

Меняет местами выбранные исходный и целевой языки, если `From` задан явно и языки различаются. Если `From` установлен в `auto` или языки совпадают, кнопка возвращает пару по умолчанию: `From = auto`, `To = en`.

### `See translated prompts after click Generate`

**По умолчанию:** выключено.

Показывает два поля предпросмотра: `Prompt translate` и `Negative Translate`. Если флажок включён, после завершения обычной кнопки **Generate** расширение ещё раз переводит текст из основных полей и показывает результат в этих полях. Предпросмотр только отображает перевод: сам по себе он не заменяет введённый prompt.

Этот предпросмотр привязан к обычной кнопке **Generate**. Он не подключён к отдельным кнопкам запуска Prompt Batch или Image Batch. Сам автоматический перевод при этом может работать и в пакетных заданиях, если включён **`Enabled`**.

## Ручной перевод до генерации

1. Оставьте **`Enabled`** выключенным, если не хотите, чтобы текст повторно переводился в процессе генерации.
2. Выберите язык в **`From`** или оставьте `auto`.
3. Выберите целевой язык в **`To`**.
4. Заполните положительный и, если нужно, отрицательный prompt.
5. Нажмите **`Translate`**.
6. Просмотрите переведённый текст в основных полях. При необходимости отредактируйте его и нажмите **Generate**.

Этот способ удобен, когда нужно проверить перевод до отправки текста в генератор. Обратите внимание: ручной перевод перезаписывает содержимое основных полей prompt, поэтому исходный текст не останется в них автоматически.

## В каких режимах работает перевод

### Обычная генерация

При включённом **`Enabled`** положительный и отрицательный prompt переводятся в обработчике каждого задания перед дальнейшей подготовкой к генерации. При ручном нажатии **`Translate`** текст меняется сразу, даже если **`Enabled`** выключен.

### Prompt Batch

Prompt Batch запускает несколько текстовых заданий по очереди. Для каждого задания, проходящего обычную обработку prompt, включённый **`Enabled`** применяет перевод с выбранными `From` и `To`. Если используется предпросмотр, учтите, что его обновляет только цепочка обычной кнопки **Generate**, а не кнопка запуска Prompt Batch.

## Что получается в результате

Prompt Translate не создаёт отдельный файл перевода. При ручном переводе результат помещается в основные поля prompt, а также в поля предпросмотра. При автоматическом переводе текст используется внутри задания; изображение сохраняется обычным способом Fooocus Extend.

В коде явно используется `GoogleTranslator`, а не локальная модель перевода. Значит, перевод зависит от доступности внешнего сервиса Google Translate через `deep-translator` и сетевого соединения. Не отправляйте через этот режим конфиденциальные тексты, если не хотите передавать их внешнему переводчику. 

## Если перевод не сработал

**После Generate prompt остался на исходном языке.** Проверьте, включён ли **`Enabled`**. Выбор `From` и `To` сам по себе автоматический перевод не включает.

**Кнопка `Translate` заменила мои слова, а исходник нужен обратно.** Ручная кнопка записывает перевод непосредственно в основные поля. Верните исходный текст вручную либо вставьте его заново из сохранённой копии. Для проверки без замены исходника включайте **`See translated prompts after click Generate`** и используйте обычную кнопку Generate.

**Перевод не появляется или генерация завершается ошибкой.** Реализация вызывает переводчик напрямую и не содержит собственного перехвата ошибок. Проверьте доступ к интернету и доступность Google Translate; для краткого текста с неоднозначным языком попробуйте выбрать `From` вручную, а в качестве целевого — `en`.
