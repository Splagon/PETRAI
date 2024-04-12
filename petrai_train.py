from transformers import BartTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, DatasetDict, Dataset
import numpy
import evaluate

iteration = 1
from_perc, to_perc = 0.00, 1.00
src_lang = "en"
tgt_lang = "ru"
model_name = "petrai_" + src_lang + "-" + tgt_lang + "_bart_opus100-" + iteration

train_dataset = load_dataset("opus100", "en-ru")
training_len = len(train_dataset['train'])
from_index, to_index = int(training_len*from_perc), int(training_len*to_perc)
train_dataset = DatasetDict({'train': Dataset.from_dict(train_dataset['train'][from_index:to_index]), 'test': train_dataset['test']})

checkpoint = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(checkpoint)

prefix = "translate " + src_lang + " to " + tgt_lang + ": "

def preprocess_function(examples):
    inputs = [prefix + example[src_lang] for example in examples["translation"]]
    refs = [example[tgt_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=refs, max_length=128, truncation=True)
    return model_inputs
tokenized_dataset = train_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

bleu = evaluate.load("sacrebleu")

def postprocess_text(preds, refs):
    preds = [pred.strip() for pred in preds]
    refs = [[ref.strip()] for ref in refs]
    return preds, refs

def compute_metrics(eval_preds):
    preds, refs = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    refs = numpy.where(refs != -100, refs, tokenizer.pad_token_id)
    decoded_refs = tokenizer.batch_decode(refs, skip_special_tokens=True)

    decoded_preds, decoded_refs = postprocess_text(decoded_preds, decoded_refs)

    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_refs)
    metrics = {"bleu": bleu_score["score"]}

    prediction_lens = [numpy.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    metrics["gen_len"] = numpy.mean(prediction_lens)
    metrics = {metric: round(val, 4) for metric, val in metrics.items()}
    return metrics

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
modelLocation = "./translator_models/"

training_args = Seq2SeqTrainingArguments(
    output_dir=model_name + "_checkpoints",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(modelLocation+model_name)