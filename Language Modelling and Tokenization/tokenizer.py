import re
import sys

class CustomTokenizer:
    def __init__(self):
        # Define regular expressions for different token patterns
        self.patterns = [
            (r'((?<=[^A-Z].[.?!])\s[a-zA-Z][^.!?]+[.?!])|((?:^|(?<=[\"-]))[a-zA-Z][^.!?]+[.?!])', 'SENTENCE'),  # Sentence Tokenizer
            (r'((\b\w+\b)|(<\w+>))|([.,;:!?\'\"])', 'WORD'),  # Word Tokenizer
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'MAILID'),  # Mail IDs
            (r'https?://\S+|www.\S+', 'URL'),  # URLs
            (r'#\w+', 'HASHTAG'),  # Hashtags
            (r'@\w+', 'MENTION'),  # Mentions
            (r'\b\d+\b', 'NUMBER'),  # Numbers
        ]
        self.modifications = [
            (r'Mr\.', 'Mr'),
            (r'Mrs\.', 'Mrs')
        ]
    
    def text_modifications(self, text):
        for pattern, sub in self.modifications:
            text = re.sub(pattern, sub, text)
        
        return text

    def replace_placeholders(self, text):
        for pattern, label in self.patterns[2:]:
            text = re.sub(pattern, f"<{label}>", text)
        return text

    def sentence_tokenize(self, text):
        tokens = []
        pattern, label = self.patterns[0]
        matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
        tokens = [match.group() for match in matches]
        return tokens
    
    def words_tokenize(self, tokens):
        word_tokens = []
        pattern, label = self.patterns[1]
        for i, sentence in enumerate(tokens):
            matches = re.finditer(pattern, sentence)
            word_tokens.append([match.group() for match in matches])
        return word_tokens
        
    def print_sentences(self, tokens):
        for i,(token, label) in enumerate(tokens):
            if label == 'SENTENCE':
                print("Sentence",i,":",token)

    def generator_tokenizer(self, test_sentence):
        replaced_text = self.replace_placeholders(self.text_modifications(test_sentence))
        sentences = replaced_text.split('\n')
        word_tokens =[]
        for sentence in sentences:
            matches = re.finditer(self.patterns[1][0], sentence)
            word_tokens.append([match.group() for match in matches])
        return word_tokens






def tokenise(file_path, file = True):
    if file == True:
        with open(file_path) as file:
            text_to_tokenize = file.read()
    else:
        text_to_tokenize = file_path
    tokenizer = CustomTokenizer()
    sub_text = tokenizer.text_modifications(text_to_tokenize)
    replaced_text = tokenizer.replace_placeholders(sub_text)
    tokens = tokenizer.sentence_tokenize(replaced_text)
    word_tokens = tokenizer.words_tokenize(tokens)

    return word_tokens


file_path = ''
if __name__ == "__main__":
    text = input("Your text: ")
    tokenizer = CustomTokenizer()
    sub_text = tokenizer.text_modifications(text)
    replaced_text = tokenizer.replace_placeholders(sub_text)
    tokens = tokenizer.sentence_tokenize(replaced_text)
    word_tokens = tokenizer.words_tokenize(tokens)  

    print("Tokenized Text: ",word_tokens)

