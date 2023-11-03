# Introduction

We present a large-scale dataset, InViG, for interactive visual
grounding under language ambiguity. Our dataset comprises over
520K images accompanied by open-ended, goal-oriented disambiguation 
dialogues, encompassing millions of object instances and 
corresponding question-answer pairs. Leveraging the InViG dataset, 
we conduct extensive studies and propose a set of baseline solutions 
for end-to-end interactive visual disambiguation and grounding, 
achieving a 45.6\% success rate during validation. To the best of 
our knowledge, the InViG dataset is the first large-scale dataset 
for resolving open-ended interactive visual grounding, presenting a 
a practical yet highly challenging benchmark for ambiguity-aware HRI.

This repo includes APIs for reading and visualizing the InViG dataset.

## InViG 500K

### Annotation

Each line in the annotation file includes keys:

```
"filename": the image file which this annotation is for.
"width": image width
"height": image height
"ann":
    "bboxes": a list of bounding boxes from SAM dataset
    "ref_exp": the initial instruction
    "questions": questions to disambiguate
    "answers": the corresponding answers
    "ref_bboxes": the box that the dialog is about
"id": the unique id of this annotation
"label" (if applicable): the object class label. Detailed mapping can be found in the object_categories.json in this repo.
```


### Images

Images are compressed using zip into blocks. To uncompress, you need to:

```console
zip -F invig500k.zip --out invig500k_full.zip
unzip invig500k_full.zip
```

The script will first concatenate all blocks into a big file and then uncompress it.
The big file will be about 230GB.

## InViG 21K

### Annotation

Mostly, it is similar to InViG 500K. Some differences are listed here.

For training set ``invig21k_train_anns.jsonl``:

```
"invig_label_id": the globally unique id in the whole InViG 21K dataset including all splits.
"id": the unique id in the corresponding split
```

For validation set ``invig21k_valid_anns.jsonl`` and test set ``invig21k_test_anns.jsonl``:

```
"ann": 
    "question_candidates": the candidates for all questions
    "answer_candidates": the candidates for all answers
    # Each question or answer has 30 incorrect candidates.
```

## API Usage

### Preparation

Please first set your annotations and images following the file 
structure as follows:

```
IMAGE_ROOT/
    invig21k_imgs/
        xxx.jpg/png
    invig500k_imgs/
        xxx.jpg/png
ANN_ROOT/
    invig21k_anns/
        invig21k_train_anns.jsonl
        invig21k_valid_anns.jsonl
        invig21k_test_anns.jsonl
    invig500k_anns/
        invig500k_anns.jsonl
```

### Usage

To use the api, you can follow the codes:

```python
from invig_api import INVIGAPI

invig_api = INVIGAPI(YOUR_ANN_ROOT, YOUR_PATH_ROOT)

print(len(invig_api))

# get annotation
print(invig_api[10])

# get data by type
print(invig_api.get_image(10))
print(invig_api.get_bboxes(10))
print(invig_api.get_dialogues(10))

# get data for different tasks
print(invig_api.get_mvqa(10))
print(invig_api.get_mvqg(10))
print(invig_api.get_mvg(10))
```
