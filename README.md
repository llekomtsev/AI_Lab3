# Лабораторная работа 3  
## Семантический поиск по коллекции изображений с использованием CLIP и FAISS

**Дисциплина:** Проектирование систем интеллектуального анализа промышленных данных  
**Тема:** Мультимодальные модели  
**Проект:** Семантический поиск по коллекции изображений  
**Датасет:** COCO val2014  
**Модель:** CLIP  
**Векторная база:** FAISS  
**Интерфейс:** Gradio / интерактивная форма в Google Colab  
**Среда выполнения:** Google Colab  
**Фреймворк:** PyTorch  

---

## 1. Цель работы

Цель лабораторной работы — разработать систему семантического поиска по коллекции изображений с использованием мультимодальной модели **CLIP**.

В рамках работы необходимо построить поисковик, который по текстовому запросу пользователя находит наиболее релевантные изображения из коллекции.

Общий принцип работы системы:

```text
текстовый запрос
        ↓
CLIP Text Encoder
        ↓
текстовый эмбеддинг
        ↓
поиск ближайших изображений в FAISS
        ↓
Top-K релевантных изображений
````

---

## 2. Постановка задачи

По условию лабораторной работы необходимо:

1. Взять датасет из 10–50 тысяч изображений.
2. Рассчитать эмбеддинги изображений с помощью CLIP.
3. Сохранить эмбеддинги в FAISS или Qdrant.
4. Реализовать поиск по текстовому запросу.
5. Выдавать Top-K наиболее релевантных картинок.
6. Реализовать простой веб-интерфейс на Gradio или Streamlit.
7. Провести анализ результатов.

В данной работе используется следующая конфигурация:

| Компонент              | Значение                              |
| ---------------------- | ------------------------------------- |
| Датасет                | COCO val2014                          |
| Количество изображений | 20 000                                |
| Модель                 | `openai/clip-vit-base-patch32`        |
| Размерность эмбеддинга | 512                                   |
| Векторная база         | FAISS                                 |
| Тип индекса            | `IndexFlatIP`                         |
| Метрика близости       | Cosine similarity через inner product |
| Интерфейс              | Gradio + Colab widgets                |

---

## 3. Теоретическая база

### 3.1. Семантический поиск изображений

Семантический поиск изображений — это поиск не по имени файла или тегам, а по смыслу запроса.

Например, если пользователь вводит запрос:

```text
a dog playing with a ball
```

система должна найти изображения, на которых действительно есть собака, играющая с мячом, даже если в имени файла нет слов `dog` или `ball`.

Для этого изображения и текстовые запросы переводятся в общее векторное пространство.

---

### 3.2. Мультимодальная модель CLIP

**CLIP** — это мультимодальная модель, которая умеет сопоставлять изображения и текст.

CLIP состоит из двух основных частей:

| Компонент     | Назначение                                      |
| ------------- | ----------------------------------------------- |
| Image Encoder | преобразует изображение в вектор признаков      |
| Text Encoder  | преобразует текстовый запрос в вектор признаков |

В данной работе используется модель:

```text
openai/clip-vit-base-patch32
```

CLIP преобразует каждое изображение в вектор размерности 512:

```text
image → CLIP Image Encoder → image embedding
```

Текстовый запрос также преобразуется в вектор размерности 512:

```text
text query → CLIP Text Encoder → text embedding
```

После этого можно сравнивать текстовый эмбеддинг с эмбеддингами изображений.

---

### 3.3. Эмбеддинги

Эмбеддинг — это числовой вектор, который описывает смысл объекта.

В данной работе используются:

| Объект           | Эмбеддинг       |
| ---------------- | --------------- |
| Изображение      | image embedding |
| Текстовый запрос | text embedding  |

Если текстовый запрос и изображение близки по смыслу, их эмбеддинги будут близки в векторном пространстве.

Перед добавлением в FAISS эмбеддинги нормализуются:

```python
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
```

После нормализации inner product эквивалентен cosine similarity.

---

### 3.4. FAISS

**FAISS** — библиотека для быстрого поиска ближайших векторов.

В данной работе используется индекс:

```python
faiss.IndexFlatIP(embedding_dim)
```

где:

* `IndexFlatIP` — точный поиск по inner product;
* `embedding_dim = 512`;
* в индекс добавляются CLIP-эмбеддинги изображений.

Схема работы FAISS:

```text
image embeddings
        ↓
FAISS index
        ↓
text embedding
        ↓
поиск ближайших image embeddings
        ↓
Top-K изображений
```

---

## 4. Описание разработанной системы

### 4.1. Общая схема системы

Система работает по следующему алгоритму:

```text
COCO val2014
        ↓
выбор 20 000 изображений
        ↓
CLIP Image Encoder
        ↓
нормализованные image embeddings
        ↓
FAISS index
        ↓
текстовый запрос пользователя
        ↓
CLIP Text Encoder
        ↓
нормализованный text embedding
        ↓
поиск Top-K ближайших изображений
        ↓
вывод результатов в интерфейсе
```

---

### 4.2. Загрузка датасета

В работе используется датасет **COCO val2014**.

Архив загружается из официального источника:

```python
!wget -O /content/AI_Lab3_CLIP_Image_Search/data/val2014.zip \
http://images.cocodataset.org/zips/val2014.zip
```

После распаковки формируется список изображений:

```python
image_paths = []

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_paths.append(os.path.join(IMAGE_DIR, file))
```

Из всего набора выбирается 20 000 изображений:

```python
NUM_IMAGES = 20000
selected_image_paths = random.sample(image_paths, min(NUM_IMAGES, len(image_paths)))
```

---

### 4.3. Примеры изображений из датасета

Для проверки загрузки данных были выведены несколько изображений из выбранной коллекции.

<img width="2065" height="1007" alt="image" src="https://github.com/user-attachments/assets/1027f9a5-b1e9-459e-9674-6992e8800d05" />


---

### 4.4. Расчёт эмбеддингов изображений

Для каждого изображения вычисляется CLIP-эмбеддинг.

Функция обработки изображений:

```python
@torch.no_grad()
def get_image_embeddings(image_paths, batch_size=32):
    all_embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]

        images = []
        current_valid_paths = []

        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            images.append(image)
            current_valid_paths.append(path)

        inputs = clip_processor(
            images=images,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = clip_model.get_image_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            image_features = outputs.pooler_output
        else:
            image_features = outputs

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu().numpy())
        valid_paths.extend(current_valid_paths)

    embeddings = np.vstack(all_embeddings).astype("float32")

    return embeddings, valid_paths
```

Результат:

```text
Image embeddings shape: (20000, 512)
Valid image paths: 20000
```

---

### 4.5. Создание FAISS-индекса

После расчёта эмбеддингов создаётся FAISS-индекс:

```python
embedding_dim = image_embeddings.shape[1]

index = faiss.IndexFlatIP(embedding_dim)
index.add(image_embeddings)
```

Размер индекса:

```text
FAISS index size: 20000
```

Также индекс и пути к изображениям сохраняются на диск:

```python
faiss.write_index(index, faiss_index_path)

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(valid_image_paths, f, ensure_ascii=False, indent=2)
```

Сохраняемые файлы:

```text
clip_coco_faiss.index
image_paths.json
```

---

### 4.6. Поиск по текстовому запросу

Текстовый запрос пользователя преобразуется в CLIP-эмбеддинг:

```python
@torch.no_grad()
def get_text_embedding(query):
    inputs = clip_processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)

    outputs = clip_model.get_text_features(**inputs)

    if hasattr(outputs, "pooler_output"):
        text_features = outputs.pooler_output
    else:
        text_features = outputs

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().astype("float32")
```

Далее выполняется поиск Top-K ближайших изображений:

```python
def search_images(query, top_k=5):
    query_embedding = get_text_embedding(query)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        image_path = valid_image_paths[idx]
        results.append((image_path, float(score)))

    return results
```

---

## 5. Интерфейс системы

### 5.1. Gradio-интерфейс

Для демонстрации работы поисковика был реализован интерфейс на **Gradio**.

Код интерфейса:

```python
def gradio_search(query, top_k):
    results = search_images(query, top_k=int(top_k))

    output = []

    for path, score in results:
        image = Image.open(path).convert("RGB")
        caption = f"Score: {score:.3f}\n{os.path.basename(path)}"
        output.append((image, caption))

    return output

demo = gr.Interface(
    fn=gradio_search,
    inputs=[
        gr.Textbox(
            label="Text query",
            placeholder="Example: a dog playing with a ball"
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Top-K"
        )
    ],
    outputs=gr.Gallery(
        label="Search results",
        columns=5,
        height="auto"
    ),
    title="CLIP Semantic Image Search",
    description="Semantic image search over COCO images using CLIP embeddings and FAISS."
)
```

Из-за особенностей запуска в Google Colab публичная ссылка Gradio может работать нестабильно. Поэтому дополнительно был реализован интерактивный интерфейс прямо в Colab.

---

### 5.2. Интерактивная форма в Google Colab

Для локальной демонстрации работы поиска в Colab была создана форма с полем ввода запроса, слайдером Top-K и кнопкой поиска.

<img width="1206" height="142" alt="image" src="https://github.com/user-attachments/assets/fb3c463e-5fdf-4da7-aad4-4f8bb03be1bf" />


Интерфейс позволяет:

* ввести текстовый запрос;
* выбрать количество результатов Top-K;
* нажать кнопку `Search`;
* получить найденные изображения с оценками similarity.

---

## 6. Результаты поиска

Для проверки работы системы были использованы следующие текстовые запросы:

```text
a dog playing with a ball
people riding bicycles
food on a table
a red bus on the street
a cat sitting on a sofa
```

---

### 6.1. Запрос: `a dog playing with a ball`

<img width="1363" height="765" alt="image" src="https://github.com/user-attachments/assets/97a5aad1-1207-4111-89f4-a9b25a9c0921" />


По запросу модель возвращает изображения, содержащие собак, игры, мячи или близкий визуальный контекст.

---

### 6.2. Запрос: `people riding bicycles`

<img width="1393" height="757" alt="image" src="https://github.com/user-attachments/assets/6aa9adcd-9516-4a30-af4c-96a1416622c0" />


По запросу модель находит изображения с людьми на велосипедах или сценами, связанными с ездой на велосипеде.

---

### 6.3. Запрос: `food on a table`

<img width="1363" height="755" alt="image" src="https://github.com/user-attachments/assets/f9537df8-667e-4a4a-9f99-dbebe1f2bf47" />


По запросу модель выдаёт изображения с едой, тарелками, столами и сценами приёма пищи.

---

### 6.4. Запрос: `a red bus on the street`

<img width="1362" height="782" alt="image" src="https://github.com/user-attachments/assets/b38578f4-9b82-42ce-b1e1-4ba5a9d646ae" />


По запросу модель ищет изображения с автобусами, городскими дорогами и транспортом.

---

### 6.5. Запрос: `a cat sitting on a sofa`

<img width="1370" height="767" alt="image" src="https://github.com/user-attachments/assets/66ef915e-aeb5-4b69-a36e-f3c6484b2047" />


По запросу модель возвращает изображения с кошками, домашней обстановкой и похожими сценами.

---

## 7. Анализ similarity score

Так как задача является задачей семантического поиска без ручной разметки релевантности для каждого запроса, классические метрики классификации, такие как Accuracy, Precision, Recall и F1-score, не применялись.

Для анализа использовались:

* CLIP cosine similarity;
* сравнение Top-1, Top-5 и Top-10 similarity;
* распределение similarity score;
* время поиска FAISS;
* визуальная оценка Top-K выдачи.

---

### 7.1. Таблица similarity score

Для каждого тестового запроса были рассчитаны:

| Показатель         | Описание                            |
| ------------------ | ----------------------------------- |
| `top1_score`       | similarity score лучшего результата |
| `top5_mean_score`  | среднее similarity score для Top-5  |
| `top10_mean_score` | среднее similarity score для Top-10 |
| `min_top10_score`  | минимальный score среди Top-10      |
| `max_top10_score`  | максимальный score среди Top-10     |

Таблица сохраняется в файл:

```text
search_metrics_summary.csv
```
<img width="1005" height="226" alt="image" src="https://github.com/user-attachments/assets/0c0174b3-563f-4e22-927e-a7864c3ba3ab" />

---

### 7.2. Графики similarity score по рангу

Для каждого запроса были построены графики изменения similarity score от позиции результата в выдаче.

Графики:

<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/9a0817f0-1ac4-4418-9e86-b699681b0162" />
<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/e956613f-6be9-4671-a4a0-efbf560e86cb" />
<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/167f5d8c-1bfa-4965-923c-6bcd414457b1" />
<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/f8de51da-7ed3-4508-ab49-c9a71afdf59f" />
<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/fd60ff54-220a-4a0e-8d65-88484e2df7da" />


На графиках видно, что первые результаты имеют наибольшую близость к запросу, а по мере увеличения ранга similarity score постепенно снижается.

---

### 7.3. Сравнение Top-1, Top-5 и Top-10 similarity

<img width="979" height="590" alt="image" src="https://github.com/user-attachments/assets/b1dd308a-c704-4457-b306-fb80f5bb8363" />

На графике сравниваются значения:

* Top-1 similarity;
* средний Top-5 similarity;
* средний Top-10 similarity.

Top-1 score обычно выше средних значений Top-5 и Top-10, что логично: первый результат является наиболее близким к текстовому запросу.

---

### 7.4. Распределение similarity score

Для запроса:

```text
a dog playing with a ball
```

было построено распределение cosine similarity по всей базе изображений.

<img width="1580" height="977" alt="similarity_distribution_dog_query" src="https://github.com/user-attachments/assets/0e9382bf-8b7c-40a9-bf99-4d5decc9743d" />


Большинство изображений имеет сравнительно невысокую близость к запросу. Top-K результаты находятся в верхней части распределения, поэтому возвращаются системой как наиболее релевантные.

---

## 8. Анализ времени поиска

Для каждого тестового запроса поиск выполнялся несколько раз, после чего вычислялось среднее время поиска.

Результаты сохраняются в файл:

```text
search_time_summary.csv
```
<img width="1033" height="268" alt="image" src="https://github.com/user-attachments/assets/3f230352-b8f7-4b16-bf18-1115e3b37af0" />


График среднего времени поиска:

<img width="982" height="490" alt="image" src="https://github.com/user-attachments/assets/4bb4019c-44ef-44bc-a5eb-16649a6a919d" />


Использование FAISS позволяет выполнять поиск быстро, так как сравнение происходит между векторами в индексе, а не между исходными изображениями.

---

## 9. Обсуждение результатов

Тестирование запроса в коллаб:
<img width="1777" height="552" alt="image" src="https://github.com/user-attachments/assets/3c20081e-f75a-435f-b7d9-7608470a1ae2" />


В ходе работы была реализована система семантического поиска по изображениям.

Основные наблюдения:

1. CLIP позволяет сопоставлять текст и изображения в общем векторном пространстве.
2. FAISS обеспечивает быстрый поиск ближайших изображений среди 20 000 эмбеддингов.
3. Система способна находить релевантные изображения по естественным текстовым запросам.
4. Запросы с конкретными объектами, например `dog`, `bus`, `cat`, дают хорошо интерпретируемую выдачу.
5. Чем ниже позиция результата в Top-K, тем обычно ниже similarity score.
6. Интерфейс позволяет менять текстовый запрос и количество возвращаемых результатов.
7. Для демонстрации в Colab использована интерактивная форма, так как внешний Gradio-туннель может быть нестабилен.

---

## 10. Выводы

В лабораторной работе была разработана система семантического поиска по коллекции изображений с использованием CLIP и FAISS.

Были выполнены следующие этапы:

1. Загружен датасет COCO val2014.
2. Из датасета выбрана коллекция из 20 000 изображений.
3. Для каждого изображения рассчитан CLIP image embedding.
4. Эмбеддинги нормализованы и добавлены в FAISS index.
5. Реализован поиск по текстовому запросу.
6. Текстовый запрос преобразуется в CLIP text embedding.
7. FAISS возвращает Top-K наиболее близких изображений.
8. Реализован интерфейс на Gradio.
9. Дополнительно реализована интерактивная форма в Google Colab.
10. Проведён анализ similarity score и времени поиска.

Итоговый pipeline:

```text
Text query
    ↓
CLIP Text Encoder
    ↓
Text embedding
    ↓
FAISS search
    ↓
Top-K image embeddings
    ↓
Relevant images
```

Таким образом, была реализована полноценная мультимодальная система, которая выполняет семантический поиск изображений по текстовому описанию.

---

## 11. Использованные источники

1. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
2. [OpenAI CLIP](https://github.com/openai/CLIP)
3. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
4. [FAISS Documentation](https://faiss.ai/)
5. [COCO Dataset](https://cocodataset.org/)
6. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
7. [Gradio Documentation](https://www.gradio.app/docs)


## 12. Запуск проекта

Для запуска проекта необходимо открыть notebook:

```text
AI_Lab3.ipynb
```

и последовательно выполнить все ячейки.

Основные зависимости:

```text
torch
torchvision
transformers
faiss-cpu
gradio
pillow
tqdm
matplotlib
pandas
numpy
ipywidgets


