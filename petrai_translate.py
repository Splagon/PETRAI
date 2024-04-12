from translator_models import PETRAI

seperator = "--------------------------------------------"
print(seperator+"\n             WELCOME TO PETRAI\n\n      Proficient English To Russian AI\n"+seperator+"\n")
print("Press Ctrl + D to quit the program\n")

petrai = PETRAI()
langs = ["en", "ru"]

while True:
    src_lang = ""
    while (src_lang not in langs):
        src_lang = input("choose input language ("+ langs[0] + "/" + langs[1] + "): ")
        if (src_lang not in langs):
            print('## INCORRECT INPUT ##')
    src_text = input("text to translate: ")

    langs.remove(src_lang)
    tgt_lang = langs[0]
    tgt_text = petrai.translate(src_text, src_lang, tgt_lang)
    
    print("\noutput: " + tgt_text)
    langs.append(src_lang)
    print("\n"+seperator+"\n")
