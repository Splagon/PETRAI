from translator_models import PETRAI

def test(name, result):
    if result:
        return True
    else:
        raise Exception(name + " test case has failed")

def can_split_into_sentences(model, text, expected_result):
    split_text = model.split_into_sentences(text)
    return test("can_split_into_sentences", len(split_text)==expected_result)

def test_can_split_into_sentences_acceptable(model):
    text = "Hello there General Kenobi. I am Ben. What is your name?"
    return can_split_into_sentences(model, text, 3)

def test_can_split_into_sentences_zero_text(model):
    text = ""
    return can_split_into_sentences(model, text, 0)

def test_can_split_into_sentences_lots_of_text(model):
    text = "hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi. hi."
    return can_split_into_sentences(model, text, 20)

def test_can_split_into_sentences_None(model):
    text = None
    try:
        can_split_into_sentences(model, text, None)
    except (Exception):
        return True
    else:
        return False

def can_translate(model, text, src, tgt, expected_results):
    translation = model.translate(text, src, tgt, log=False)
    return test("can_translate", translation in expected_results)

def test_can_translate_en_to_ru(model):
    text = "Hello."
    return can_translate(model, text, "en", "ru", ["Привет.", "Здравствуйте.", "Алло."])

def test_can_translate_ru_to_en(model):
    text = "Привет."
    return can_translate(model, text, "ru", "en", ["Hello.", "Hey.", "Hi."])

def test_can_translate_en_to_ru_no_text(model):
    text = ""
    return can_translate(model, text, "en", "ru", "")

def test_can_translate_ru_to_en_no_text(model):
    text = ""
    return can_translate(model, text, "ru", "en", "")

def can_translate_text(model, text, src, tgt):
    translation = model.translate(text, src, tgt, log=False)
    return test("can_translate", isinstance(translation, str))

def test_can_translate_text_en_to_ru(model):
    text = "Hello. I am the policeman. Hands in the air!"
    return can_translate_text(model, text, "en", "ru")

def test_can_translate_text_ru_to_en(model):
    text = "Привет. Я крутой профессор. Как вас зовут?"
    return can_translate_text(model, text, "ru", "en")

def test_model(model):
    tests = [
        test_can_split_into_sentences_acceptable(model),
        test_can_split_into_sentences_zero_text(model),
        test_can_split_into_sentences_lots_of_text(model),
        test_can_split_into_sentences_None(model),
        test_can_translate_en_to_ru(model),
        test_can_translate_en_to_ru_no_text(model),
        test_can_translate_ru_to_en_no_text(model),
        test_can_translate_ru_to_en(model),
        test_can_translate_text_en_to_ru(model),
        test_can_translate_text_ru_to_en(model),
    ]
    return False not in tests

def test_models(models):
    for model in models:
        if test_model(model):
            print(model.name + " has passed")

test_models([PETRAI()])