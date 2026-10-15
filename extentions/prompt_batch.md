## What Prompt Batch Does
([Читать на русском языке](#что-делает-prompt-batch))

Prompt Batch runs multiple generations sequentially. Each row of the table contains a positive prompt and a negative prompt. If you upload a text file, the extension converts its lines into such a table. All rows use the common generation settings from the standard Fooocus interface: the selected model, styles, quality, size, number of images per prompt, and other parameters.

## Tab Settings and Elements

### `prompt` / `negative prompt` Table

The table has two columns:

- `prompt` — the positive description of the desired image;
- `negative prompt` — what you want to avoid.

Initially, one empty row is displayed. The table is edited directly in the interface: enter prompts into the cells to create a queue. 
**Start batch** goes through the rows from top to bottom. An empty positive prompt is skipped, and generation is not launched for it.
**Delete last row** removes the last row of the table if there is more than one row. 
**Clear Batch** clears the table and returns a single empty row.

### `Base positive prompt:`

Determines how the positive prompt from the main Fooocus form combines with the `prompt` from the table row. **None** is selected by default. This field only affects the positive part of the prompt.

- **None** — the table row replaces the main positive prompt. If the row cell is empty, this queue element is not generated.
- **Prefix** — the main positive prompt is added before the row text. Example: base `portrait, soft light`, row `a woman in a garden` → `portrait, soft light a woman in a garden`.
- **Suffix** — the main positive prompt is added after the row. Example: same base and row → `a woman in a garden portrait, soft light`.

### `Base negative prompt:`

Works the same way, but for the negative part. **None** is selected by default.

- **None** — the row's negative prompt completely replaces the main Fooocus negative prompt.
- **Prefix** — the main Fooocus negative prompt is placed before the row's negative text.
- **Suffix** — the row's negative text is placed first, and the main negative prompt comes after it.

If you leave the `negative prompt` cell empty and select **None**, the main negative prompt from the main interface is not preserved. Choose **Prefix** or **Suffix** if you want to add a common negative list to all rows.

### `only positive prompts`

This checkbox is only needed when loading a file, not during normal manual table filling. Enabled by default.

- **Enabled:** every non-empty line of the file becomes a separate positive `prompt`, and the negative column remains empty. Empty lines are ignored. Choose this mode if the file is just a list of options.
- **Disabled:** lines are read in pairs: the first non-empty line of the pair is the positive prompt, the next non-empty line is the negative one. The next pair starts after these two lines. Empty lines are skipped. If the line after a positive prompt is empty, the positive prompt is still added with an empty negative one; an empty line does not reserve a separate spot for negative text.

Example for the disabled checkbox:
```
portrait of a traveler
blurry, low quality

red bicycle in the rain

```


The first entry will get `portrait of a traveler` and `blurry, low quality`. The second will get `red bicycle in the rain` and an empty negative prompt. If you don't need a negative prompt for a pair, leave an empty line after the positive prompt, and start the next prompt with the next non-empty line. With the checkbox disabled, empty lines are skipped — they are not considered independent negative lines.

### `Load prompts from file`

The button loads a **single** text file into the table. The file extension can be anything: the text structure is what matters. When reading, the application determines the encoding by the first 1000 bytes, and then loads the lines. For simplicity, use a standard `.txt` in UTF-8 encoding. If characters are displayed incorrectly or the upload fails, resave the file in UTF-8.

Before loading, select the desired **only positive prompts** value. After reading the file, the table is replaced with the obtained lines.

### `Start batch`

Launches the processing of filled rows one by one. Before generation starts, queue editing is locked so that the content does not change during processing. Notifications show the number of the current element. After completion, the link to the generation history is updated.

The **Stop** button terminates the queue after the current interruptible generation step. Results that have already been created remain saved; subsequent rows are not launched.

## How to Use

### Fill the Queue Manually

1. In the main Fooocus interface, select the model and configure the generation parameters.
2. If necessary, set a common positive prompt and negative prompt in the main form.
3. Open **Prompt Batch** and enter one pair of prompts per generation into the table.
4. For independent full prompts, leave **Base positive prompt = None** and **Base negative prompt = None**.
5. If you need a common positive prefix or suffix, select **Prefix** or **Suffix** respectively. Configure the Base negative prompt similarly.
6. Click **Start batch**. Monitor the notifications and results in the gallery.

### Load a List of Options from a Text File

1. Prepare a text file. With **only positive prompts** enabled, write each positive prompt on a new line.
2. Leave **only positive prompts** enabled, then click **Load prompts from file** and select the file.
3. Review the resulting table and correct the lines if necessary.
4. For a list of positive/negative pairs, uncheck the box. Place each positive prompt and the following negative one on separate lines; leave an empty line for empty negative text.
5. Check the base Prefix/Suffix and launch **Start batch**.

### Example: One Common Style for Different Scenes

In the main Fooocus form, set the base positive text to `editorial photography, warm natural light`. Enter separate plots into the table, for example `a ceramic vase on a wooden table` and `a small bookstore on a rainy street`. Set **Base positive prompt = Prefix** and **Base negative prompt = Prefix** if a common list of things to avoid is set in the main form. Then the common instructions will be added to each plot.

If you want to use the table row as the main plot and put the common quality tag at the end, switch **Base positive prompt** to **Suffix**.

### Get Multiple Images per Row

The **Image Number** parameter is taken from the main Fooocus form. If you select 3, each non-empty row triggers the generation of three images. This will increase the total number of outputs roughly proportionally to the number of non-empty rows. Choose a small set for testing before running a long list.

## Which Fooocus Settings Apply to Each Row

Prompt Batch does not create separate model settings for each row. Each entry uses the parameters collected for normal generation: **base/refiner model**, **Styles**, **Performance**, **Steps**, **Aspect Ratio**, **Image Number**, **Output Format**, **Filename Prefix**, **Negative Prompt**, seed, LoRA, and other available generation settings. They are set in the standard UI elements and passed to the handler before the queue is launched.

**Seed / Random.** With **Random** enabled, each subsequent prompt gets a new random seed. With Random disabled, the fixed seed you specified in the main form is used. This is convenient if you need to compare the effect of texts with the same seed, although different prompts may still yield different results.

Common settings can be changed before the queue is launched. Prompt Batch uses the text of the positive and negative prompts from the main form according to the **Base positive prompt** and **Base negative prompt** rules. The standard setting of other parameters remains common for the entire queue.

## Where Results Appear

Images appear in the standard Fooocus gallery and are saved in the main **Outputs path** directory (if the application is run with default logging settings).

The format is selected by the common **Output Format** parameter; by default, it is set by the installation configuration. The name includes an optional **Filename Prefix**, date, time, and a random suffix, and files are placed in a folder with the current date. Therefore, the results of different prompts do not overwrite each other with the same name.

## Troubleshooting

- **The file opened with unreadable characters or the upload ended with an error.** Try resaving it as `.txt` in UTF-8. The encoding is recognized by a small sample — the first 1000 bytes.
- **After loading a pair, the negative text ended up in the positive column.** Check **only positive prompts**. For pairs, uncheck the box and place positive and negative sequentially, without inserting an empty line between them.
- **The main negative text disappeared.** In **Base negative prompt = None** mode, it is replaced by the value from the row. Choose Prefix or Suffix if you need to add common text.
- **Some rows are not launched.** An empty `prompt` is skipped. Check that there is text in the positive column.
- **The queue stopped before the end.** Check if you pressed **Stop**: when stopped, the results of already completed elements remain, and subsequent entries are not launched.

## Что делает Prompt Batch
([Read in English](#what-prompt-batch-does))

Prompt Batch запускает несколько генераций последовательно. Каждая строка таблицы содержит положительный промпт и негативный промпт. Если вы загрузили текстовый файл, расширение преобразует его строки в такую таблицу. Для всех строк используются общие настройки генерации из обычного интерфейса Fooocus: выбранная модель, стили, качество, размер, число изображений на один промпт и другие параметры.

## Настройки и элементы вкладки

### Таблица `prompt` / `negative prompt`

В таблице две колонки:

- `prompt` — положительное описание желаемого изображения;
- `negative prompt` — то, чего вы хотите избежать.

Изначально отображается одна пустая строка. Таблица редактируется прямо в интерфейсе: введите промпты в ячейки, чтобы создать очередь. 
**Start batch** проходит строки сверху вниз. Пустой положительный промпт пропускается и для него генерация не запускается.
**Delete last row** удаляет последнюю строку таблицы, если строк больше одной. 
**Clear Batch** очищает таблицу и возвращает одну пустую строку.

### `Base positive prompt:`

Определяет, как положительный промпт из основной формы Fooocus сочетается с `prompt` из строки таблицы. По умолчанию выбрано **None**. Это поле влияет только на положительную часть промпта.

- **None** — строка таблицы заменяет основной положительный промпт. Если ячейка строки пустая, этот элемент очереди не генерируется.
- **Prefix** — основной положительный промпт добавляется перед текстом строки. Пример: база `portrait, soft light`, строка `a woman in a garden` → `portrait, soft light a woman in a garden`.
- **Suffix** — основной положительный промпт добавляется после строки. Пример: та же база и строка → `a woman in a garden portrait, soft light`.

### `Base negative prompt:`

Работает так же, но для негативной части. По умолчанию выбрано **None**.

- **None** — негативный промпт строки полностью заменяет основной негативный промпт Fooocus.
- **Prefix** — основной негативный промпт Fooocus ставится перед негативным текстом строки.
- **Suffix** — негативный текст строки ставится первым, а основной негативный промпт — после него.

Если в ячейке `negative prompt` оставить пустое значение и выбрать **None**, основной отрицательный промпт из главного интерфейса не сохраняется. Выберите **Prefix** или **Suffix**, если хотите добавлять общий негативный список ко всем строкам.

### `only positive prompts`

Флажок нужен только при загрузке файла, а не при обычном ручном заполнении таблицы. По умолчанию включён.

- **Включён:** каждая непустая строка файла становится отдельным положительным `prompt`, а негативная колонка остаётся пустой. Пустые строки игнорируются. Выберите этот режим, если в файле просто список вариантов.
- **Выключен:** строки читаются парами: первая непустая строка пары — положительный промпт, следующая непустая — отрицательный. Следующая пара начинается после этих двух строк. Пустые строки пропускаются. Если после положительной строки следующая строка пустая, положительный промпт всё равно добавляется с пустым негативным; пустая строка не резервирует отдельное место под отрицательный текст.

Пример для выключенного флажка:

```
portrait of a traveler
blurry, low quality

red bicycle in the rain

```

Первая запись получит `portrait of a traveler` и `blurry, low quality`. Вторая — `red bicycle in the rain` и пустой negative prompt. Если не нужен negative prompt для пары, оставьте после положительного промпта пустую строку, а следующий промпт начинайте следующей непустой строкой. При выключенном флажке пустые строки пропускаются — они не считаются самостоятельными отрицательными строками.

### `Load prompts from file`

Кнопка загружает **один** текстовый файл в таблицу. Расширение файла может быть любым: важна структура текста. При чтении приложение определяет кодировку по первым 1000 байтам, а затем загружает строки. Для простоты используйте обычный `.txt` в кодировке UTF-8. Если символы отображаются неправильно или загрузка не удалась, пересохраните файл в UTF-8.

Перед загрузкой выберите нужное значение **only positive prompts**. Таблица после чтения файла заменяется полученными строками.

### `Start batch`

Запускает обработку заполненных строк одну за другой. Перед началом генерации  редактирование очереди блокируется, чтобы содержимое не менялось во время обработки. В уведомлениях показывается номер текущего элемента. После завершения обновляется ссылка на историю генераций.

Кнопка **Stop** завершает очередь после текущего прерываемого шага генерации. Результаты, которые уже были созданы, остаются сохранёнными; последующие строки не запускаются.

## Как пользоваться

### Заполнить очередь вручную

1. В основном интерфейсе Fooocus выберите модель и настройте параметры генерации.
2. При необходимости задайте общий положительный промпт и негативный промпт в основной форме.
3. Откройте **Prompt Batch** и внесите в таблицу по одной паре промптов на каждую генерацию.
4. Для независимых полных промптов оставьте **Base positive prompt = None** и **Base negative prompt = None**.
5. Если нужен общий положительный префикс или суффикс, выберите **Prefix** или **Suffix** соответственно. Аналогично настройте Base negative prompt.
6. Нажмите **Start batch**. Следите за уведомлениями и результатами в галерее.

### Загрузить список вариантов из текстового файла

1. Подготовьте текстовый файл. При включённом **only positive prompts** напишите каждый положительный промпт с новой строки.
2. Оставьте **only positive prompts** включённым, затем нажмите **Load prompts from file** и выберите файл.
3. Просмотрите получившуюся таблицу и при необходимости исправьте строки.
4. Для списка пар positive/negative снимите флажок. Размещайте каждый положительный промпт и следующий за ним отрицательный отдельными строками; пустой негативный текст оставляйте пустой строкой.
5. Проверьте базовые Prefix/Suffix и запустите **Start batch**.

### Пример: один общий стиль для разных сцен

В основной форме Fooocus задайте базовый положительный текст `editorial photography, warm natural light`. В таблицу внесите отдельные сюжеты, например `a ceramic vase on a wooden table` и `a small bookstore on a rainy street`. Установите **Base positive prompt = Prefix** и **Base negative prompt = Prefix**, если в основной форме задан общий список того, чего следует избегать. Тогда общие указания будут добавляться к каждому сюжету.

Если вы хотите использовать строку таблицы как основной сюжет, а общий тег качества поставить в конце, переключите **Base positive prompt** на **Suffix**.

### Получить несколько картинок на одну строку

Параметр **Image Number** берётся из основной формы Fooocus. Если выбрать 3, каждая непустая строка запускает генерацию трёх изображений. Это увеличит общее число выходов примерно пропорционально количеству непустых строк. Выбирайте небольшой набор для проверки, прежде чем запускать длинный список.

## Какие настройки Fooocus применяются к каждой строке

Prompt Batch не создаёт отдельные настройки модели для каждой строки. Каждая запись использует параметры, собранные для обычной генерации: **base/refiner model**, **Styles**, **Performance**, **Steps**, **Aspect Ratio**, **Image Number**, **Output Format**, **Filename Prefix**, **Negative Prompt**, seed, LoRA и остальные доступные настройки генерации. Они задаются в стандартных элементах UI и передаются обработчику до запуска очереди.

**Seed / Random.** При включённом **Random** каждый следующий промпт получает новый случайный seed. При выключенном Random используется фиксированный seed, который вы указали в основной форме. Это удобно, если нужно сравнить влияние текстов при одинаковом seed, хотя разные промпты всё равно могут давать отличающиеся результаты.

Общие настройки можно менять до запуска очереди. Текст положительного и негативного промптов из основной формы Prompt Batch использует по правилам **Base positive prompt** и **Base negative prompt**. Обычная настройка остальных параметров остаётся общей для всей очереди.

## Где появляются результаты

Изображения появляются в обычной галерее Fooocus и сохраняются в основном каталоге **Outputs path** (если приложение запущено с настройками логирования по умолчанию).

Формат выбирается общим параметром **Output Format**; по умолчанию он задаётся конфигурацией установки. Имя включает необязательный **Filename Prefix**, дату, время и случайный суффикс, а файлы размещаются в папке с текущей датой. Поэтому результаты разных промптов не перезаписывают друг друга по одному и тому же имени.

## Если что-то пошло не так

- **Файл открылся нечитаемыми символами или загрузка завершилась ошибкой.** Попробуйте пересохранить его как `.txt` в UTF-8. Кодировка распознаётся по небольшой выборке — первым 1000 байтам.
- **После загрузки пары отрицательный текст попал в положительный столбец.** Проверьте **only positive prompts**. Для пар снимите флажок и располагайте positive и negative последовательно, не вставляя пустую строку между ними.
- **Основной негативный текст пропал.** В режиме **Base negative prompt = None** его заменяет значение из строки. Выберите Prefix или Suffix, если надо добавить общий текст.
- **Некоторые строки не запускаются.** Пустой `prompt` пропускается. Проверьте, что в положительной колонке есть текст.
- **Очередь остановилась раньше конца.** Проверьте, не нажали ли вы **Stop**: при остановке остаются результаты уже завершённых элементов, а следующие записи не запускаются.
