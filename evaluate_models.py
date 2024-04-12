from translator_models import PETRAI, Petrai_2, PrefixedPetrai_2, Petrai_3, PrefixedPetrai_3, M4T, FAIR, Helsinki
from torchmetrics.text import CHRFScore
from transformers import logging
from bert_score import BERTScorer
from datetime import datetime
from statistics import mean
import evaluate
import time
import csv

predictionsLocation = "./predictions/"
scoresLocation = "./scores/"

def get_translator_model(model):
    match(model):
        case("petrai"):
            return PETRAI()
        case("petrai_2"):
            return Petrai_2()
        case("prefixed_petrai_2"):
            return PrefixedPetrai_2()
        case("petrai_3"):
            return Petrai_3()
        case("prefixed_petrai_3"):
            return PrefixedPetrai_3()
        case("m4t"):
            return M4T()
        case("fair"):
            return FAIR()
        case("helsinki"):
            return Helsinki()

def translate_texts(texts, src, tgt, model, is_logging=False):
    timings = {}
    if is_logging:
        print(f'Loading Model: {model}...', end='\r')
    
    start_time = time.time()
    translator = get_translator_model(model)
    end_time = time.time()-start_time
    timings["load_translator"] = end_time

    translated_texts = []
    start_time = time.time()
    for i in range(0,len(texts)):
        if is_logging:
            print(f'Model: {model} | {src}-{tgt} | Items translated: {i}/{len(texts)}', end='\r')
        translated_texts.append(translator.translate(texts[i], src, tgt, False))
    if is_logging:
        print(f'Model: {model} | {src}-{tgt} | TRANSLATION COMPLETE           ')
    end_time = time.time()-start_time
    timings["translate_dataset"] = end_time

    return translated_texts, timings

def translate(text, src, tgt, model):
    return translate_texts([text], src, tgt, model)[0]

def prepare_test_data(src, tgt, num_of_test_data=None):
    en_test_data_source = "data/newstest2014-ruen-src.en.txt"
    ru_test_data_source = "data/newstest2014-ruen-src.ru.txt"

    with open(en_test_data_source) as en_file:
        with open(ru_test_data_source) as ru_file:
            en_lines = en_file.read().splitlines() 
            ru_lines = ru_file.read().splitlines() 
    
    if src[:2] == "en":
        inputs = en_lines
        unformatted_references = ru_lines
    else:
        inputs = ru_lines
        unformatted_references = en_lines

    predictions = list.copy(inputs)

    formatted_references = []
    for ref in unformatted_references:
        formatted_references.append([ref])

    if num_of_test_data == None:
        num_of_test_data = len(inputs)

    predictions = predictions[:num_of_test_data]
    inputs = inputs[:num_of_test_data]
    collated_references = {
        "formatted_references": formatted_references[:num_of_test_data],
        "unformatted_references": unformatted_references[:num_of_test_data]
    }

    return inputs, predictions, collated_references

def get_corpus_predictions(model, src, tgt, num_of_test_data=None):
    inputs, predictions, references = prepare_test_data(src, tgt, num_of_test_data)
    predictions, timings = translate_texts(predictions, src, tgt, model, True)
    return inputs, predictions, references, timings

def extract_tensor_value(t):
    if len(t.shape) > 0:
        return [val.item() for val in t]
    else:
        return t.item()

def get_bleu_score(translations, references):
    bleu = evaluate.load("sacrebleu")
    bleu_score = bleu.compute(predictions=translations, references=references)
    return {"score": bleu_score.get("score"), "bp": bleu_score.get("bp")}

def get_bert_score(translations, references, tgt):
    bert = BERTScorer(model_type='bert-base-multilingual-cased') 
    p,r,f = bert.score(translations, references)
    bert_score = {
        "f1": extract_tensor_value(f), 
        "precision": extract_tensor_value(p), 
        "recall": extract_tensor_value(r)
    }
    average_bertscore = {
        "f1": mean(bert_score.get("f1")), 
        "precision": mean(bert_score.get("precision")), 
        "recall": mean(bert_score.get("recall"))
    }
    return average_bertscore

def get_chrf_score(translations, references):
    chrf = CHRFScore()
    return extract_tensor_value(chrf(translations, references))

def score_model(translations, references, tgt):
    model_scores = {}
    print(f'Scoring...              ', end='\r')
    model_scores["bleu"] = get_bleu_score(translations, references.get("formatted_references"))
    model_scores["bert"] = get_bert_score(translations, references.get("unformatted_references"), tgt)
    model_scores["chrf"] = get_chrf_score(translations, references.get("formatted_references"))
    print(f'SCORING COMPLETE',)
    return model_scores

models = {
    "petrai_2":          {"notes": "3003"},
    "prefixed_petrai_2": {"notes": "3003"},
    "petrai_3":          {"notes": "3003"},
    "prefixed_petrai_3": {"notes": "3003"}, 
    "m4t":               {"notes": "3003"},
    "fair":              {"notes": "3003"},
    "helsinki":          {"notes": "3003"}
}

def evaluate_models():
    model_scores = {}
    for model_key in models.keys():
        lang_pairs = {"eng": "rus", "rus": "eng"}
        for src in lang_pairs:
            tgt = lang_pairs.get(src)
            model = model_key + "_" + src + "-" + tgt
            tested_models = ["petrai_2_eng-rus", "petrai_2_rus-eng","prefixed_petrai_2_eng-rus","prefixed_petrai_2_rus-eng","petrai_3_eng-rus", "petrai_3_rus-eng","prefixed_petrai_3_eng-rus","prefixed_petrai_3_rus-eng"]
            if model not in tested_models:
                inputs, predictions, references, timings = get_corpus_predictions(model_key, src, tgt)
                
                with open(predictionsLocation+model+"_predictions.csv", 'w', newline='') as predictionsFile:
                    writer = csv.writer(predictionsFile)
                    writer.writerow(["inputs","predictions","references"])
                    refs = references.get("unformatted_references")
                    for i in range(0,len(predictions)):
                        writer.writerow([inputs[i],predictions[i],refs[i]])
                
                model_scores[model] = score_model(predictions, references, tgt)
                
                with open(scoresLocation+"model_scores.csv", "a") as modelScoresFile:
                    writer = csv.writer(modelScoresFile)
                    scores = model_scores[model]
                    writer.writerow([model, scores["bleu"], scores["bert"], scores["chrf"], datetime.now(),models[model_key]["notes"]])
    return model_scores

def test_model_speed():
    model_timings = {}
    for model_key in models.keys():
        lang_pairs = {"eng": "rus", "rus": "eng"}
        for src in lang_pairs:
            tgt = lang_pairs.get(src)
            model = model_key + "_" + src + "-" + tgt
            tested_models = []
            if model not in tested_models:
                sample_num = 10
                translations_num = 50

                mean_timings = {}
                translator_timings = []
                translation_timings = []

                for i in range(0, sample_num):
                    inputs, predictions, references, timings = get_corpus_predictions(model_key, src, tgt, translations_num)
                    translator_timings.append(timings["load_translator"])
                    translation_timings.append(timings["translate_dataset"])

                mean_timings["load_translator"] = mean(translator_timings)
                mean_timings["translate_dataset"] = mean(translation_timings)

                model_timings[model] = mean_timings
                
                with open(scoresLocation+"model_times.csv", "a") as modelTimesFile:
                    writer = csv.writer(modelTimesFile)
                    timings = model_timings[model]
                    seconds_per_text = timings["translate_dataset"] / translations_num
                    writer.writerow([model, timings["load_translator"],timings["translate_dataset"], translations_num, seconds_per_text, sample_num, datetime.now(), models[model_key]["notes"]])
    return model_timings

#print(translate("Kush is making burgers downstairs. Misha is happy. I am eating food! Who is he?", "eng", "rus", "petrai"))
#print(translate("I am eating food!", "eng", "rus", "m4t"))

#print(translate("Я люблю тебя до луны и обратно всем сердцем, Кара.", "rus", "eng", "m4t"))

#print(translate("I love you, Cara", "eng", "rus", "prefixed_petrai_3"))
#print(translate("I love you, Cara", "eng", "rus", "bart"))

#print(translate("I am eating food!", "eng", "rus", "madlad"))
#print(translate("Апельсины — это круто, но бактериям они не нравятся.", "rus", "eng", "prefixed_petrai_3"))
#print(translate("Hi, my name is Misha. What the fuck are you saying?", "eng", "rus"))
#print(evaluate_models())
print(test_model_speed())