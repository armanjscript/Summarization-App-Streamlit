import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class mT5:
    def __init__(self):
        source = "csebuetnlp/mT5_multilingual_XLSum"
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source)
    
    @staticmethod
    def whitespace_handler(k):
        return re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))
    
    def run(self, text):
        input_ids = self.tokenizer([self.whitespace_handler(text)], return_tensors="pt", padding="max_length", truncation=True, max_length=512)["input_ids"]
        output_ids = self.model.generate(input_ids=input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokeniztion_spaces=False)
        return summary