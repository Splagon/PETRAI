from transformers import pipeline, logging, AutoProcessor, SeamlessM4Tv2Model
from nltk.tokenize import sent_tokenize

logging.set_verbosity(50)

class EnRuTranslatorModel:
    def __init__(self):
        self.symbols = ['.', '!', '?', ';', ':']
        self.initialise_translators()

    def get_pipeline(self, src, tgt):
        model_translators_dir = "./translator_models/"
        if (src[:2] == "en" and tgt[:2] == "ru"):
            model_name = self.en_ru_model_name
        elif (src[:2] == "ru" and tgt[:2] == "en"):
            model_name = self.ru_en_model_name
        else:
            raise Exception(" ** TRANSLATOR NOT FOUND ** ")
        
        pipe = pipeline("translation", src_lang=src, tgt_lang=tgt, model=model_translators_dir+model_name, max_length=1000)        
        return pipe
        
    def initialise_translators(self):
        print(f'Initialising ' + self.name + '...', end='\r')
        self.en_ru_translator = self.get_pipeline("en", "ru")
        self.ru_en_translator = self.get_pipeline("ru", "en")
        print(f"                                 ", end='\r')

    def choose_translator(self, src, tgt):
        if (src[:2] == "en" and tgt[:2] == "ru"):
            return self.en_ru_translator
        elif (src[:2] == "ru" and tgt[:2] == "en"):
            return self.ru_en_translator
        else:
            raise Exception(" ** TRANSLATOR NOT FOUND ** ")
    
    def split_into_sentences(self, text):
        if isinstance(text, str):
            sentences = sent_tokenize(text)
            return sentences
        else:
            raise Exception(" ** TRANSLATOR CAN ONLY TRANSLATE TEXT ** ")
    
    def translate_single_sentence(self, sentence, translator):
        return translator(sentence)[0].get('translation_text')

    def translate_text(self, text, translator):
        sentences = self.split_into_sentences(text)
        translated_sentences = [self.translate_single_sentence(sentence, translator) for sentence in sentences]
        translated_text = " ".join(translated_sentences)
        return translated_text
    
    def translate(self, text, src, tgt, log=True):
        if log == True:
            print(f'Translating...', end='\r')
        translator = self.choose_translator(src.lower(), tgt.lower())
        translated_text = self.translate_text(text, translator)
        if log == True:
            print(f'              ', end='\r')
        return translated_text
    
class Generic(EnRuTranslatorModel):
    def __init__(self, name, model_name):
        self.name = name

        if isinstance(model_name, tuple):
            self.en_ru_model_name = model_name[0]
            self.ru_en_model_name = model_name[1]
        else:
            self.en_ru_model_name = model_name
            self.ru_en_model_name = model_name
        super().__init__()


class PETRAI(EnRuTranslatorModel):
    def __init__(self):
        self.name = "PETRAI"
        self.en_ru_model_name = "petrai_en-ru_bart_opus100-2"
        self.ru_en_model_name = "petrai_ru-en_bart_opus100-3"
        super().__init__()

    def choose_translator(self, src, tgt):
        self.prefix = "translate " + src[:2] + " to " + tgt[:2] + ": "
        return super().choose_translator(src, tgt)

    def translate_single_sentence(self, sentence, translator):
        return super().translate_single_sentence(self.prefix+sentence, translator)

class Petrai_2(EnRuTranslatorModel):
    def __init__(self):
        self.name = "PETRAI_2"
        self.en_ru_model_name = "petrai_en-ru_bart_opus100-2"
        self.ru_en_model_name = "petrai_ru-en_bart_opus100-2"
        super().__init__()

class PrefixedPetrai_2(Petrai_2):
    def __init__(self):
        super().__init__()

    def choose_translator(self, src, tgt):
        self.prefix = "translate " + src[:2] + " to " + tgt[:2] + ": "
        return super().choose_translator(src, tgt)

    def translate_single_sentence(self, sentence, translator):
        return super().translate_single_sentence(self.prefix+sentence, translator)

class Petrai_3(EnRuTranslatorModel):
    def __init__(self):
        self.name = "PETRAI_3"
        self.en_ru_model_name = "petrai_en-ru_bart_opus100-3"
        self.ru_en_model_name = "petrai_ru-en_bart_opus100-3"
        super().__init__()

class PrefixedPetrai_3(Petrai_3):
    def __init__(self):
        super().__init__()

    def choose_translator(self, src, tgt):
        self.prefix = "translate " + src[:2] + " to " + tgt[:2] + ": "
        return super().choose_translator(src, tgt)

    def translate_single_sentence(self, sentence, translator):
        return super().translate_single_sentence(self.prefix+sentence, translator)

class M4T(EnRuTranslatorModel):
    def __init__(self):
        self.name = "M4T"
        self.model_name = "facebook/seamless-m4t-v2-large"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SeamlessM4Tv2Model.from_pretrained(self.model_name)

    def translate_single_sentence(self, sentence, src, tgt):
        text_inputs = self.processor(text=sentence, src_lang=src, return_tensors="pt")
        output_tokens = self.model.generate(**text_inputs, tgt_lang=tgt, generate_speech=False)
        translated_text_from_text = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        return translated_text_from_text

    def translate_text(self, text, src, tgt):
        sentences = self.split_into_sentences(text)
        translated_sentences = [self.translate_single_sentence(sentence, src, tgt) for sentence in sentences]
        translated_text = " ".join(translated_sentences)
        return translated_text

    def translate(self, text, src, tgt, log=True):
        translated_text = self.translate_text(text, src, tgt)
        return translated_text
    
class FAIR(EnRuTranslatorModel):
    def __init__(self):
        self.name = "FAIR"
        self.en_ru_model_name = "facebook/wmt19-en-ru"
        self.ru_en_model_name = "facebook/wmt19-ru-en"
        super().__init__()
    
class Helsinki(EnRuTranslatorModel):
    def __init__(self):
        self.name = "Opus-MT"
        self.en_ru_model_name = "Helsinki-NLP/opus-mt-en-ru"
        self.ru_en_model_name = "Helsinki-NLP/opus-mt-ru-en"
        super().__init__()