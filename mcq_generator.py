class MCQGenerator:
    def __init__(self):
        self.summarizer = None
        self.nltk = None
        self.string = None
        self.re = None
        self.random = None
        self.csv = None
        self.pke = None
        self.sent_tokenize = None
        self.stopwords = None
        self.max_similarity = None
        self.adapted_lesk = None
        self.wn = None
        self.requests = None

    def get_summarizer(self):
        if self.summarizer is None:
            from summarizer import Summarizer
            self.summarizer = Summarizer()
        return self.summarizer

    def get_nltk(self):
        if self.nltk is None:
            import nltk
            self.nltk = nltk
            if not nltk.download('stopwords', quiet=True):
                nltk.corpus.stopwords.ensure_loaded()
            if not nltk.download('popular', quiet=True):
                pass  # already downloaded
        return self.nltk

    def get_string(self):
        if self.string is None:
            import string
            self.string = string
        return self.string

    def get_re(self):
        if self.re is None:
            import re
            self.re = re
        return self.re

    def get_random(self):
        if self.random is None:
            import random
            self.random = random
        return self.random

    def get_csv(self):
        if self.csv is None:
            import csv
            self.csv = csv
        return self.csv

    def get_pke(self):
        if self.pke is None:
            import pke
            self.pke = pke
        return self.pke

    def get_sent_tokenize(self):
        if self.sent_tokenize is None:
            from nltk.tokenize import sent_tokenize
            self.sent_tokenize = sent_tokenize
        return self.sent_tokenize

    def get_stopwords(self):
        if self.stopwords is None:
            nltk = self.get_nltk()
            from nltk.corpus import stopwords
            self.stopwords = stopwords
        return self.stopwords

    def get_max_similarity(self):
        if self.max_similarity is None:
            from pywsd.similarity import max_similarity
            self.max_similarity = max_similarity
        return self.max_similarity

    def get_adapted_lesk(self):
        if self.adapted_lesk is None:
            from pywsd.lesk import adapted_lesk
            self.adapted_lesk = adapted_lesk
        return self.adapted_lesk

    def get_wn(self):
        if self.wn is None:
            from nltk.corpus import wordnet as wn
            self.wn = wn
        return self.wn

    def get_requests(self):
        if self.requests is None:
            import requests
            self.requests = requests
        return self.requests

    def get_user_input(self):
        return input("Enter the text to summarize: ")

    def summarize_text(self, full_text):
        return self.get_summarizer()(full_text, min_length=60, max_length=500, ratio=0.4)

    def tokenize_sentences(self, text):
        sentences = self.get_sent_tokenize()(text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def get_sentences_for_keyword(self, keywords, sentences):
        keyword_sentences = {}
        for word in keywords:
            keyword_sentences[word] = [sentence for sentence in sentences if word in sentence]
        return keyword_sentences

    def get_nouns_multipartite(self, text):
        out = []
        extractor = self.get_pke().unsupervised.MultipartiteRank()
        extractor.load_document(input=text)
        pos = {'PROPN', 'NOUN'}
        stoplist = set(self.get_string().punctuation)
        stoplist |= {'-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'}
        stoplist |= set(self.get_stopwords().words('english'))
        additional_stopwords = {'example', 'examples', 'task', 'entity', 'data', 'use', 'type', 'concepts', 'concept',
                                'learn', 'function', 'method', 'unit', 'fontionality', 'behavior', 'simple', 'ways',
                                'capsule', 'capsules', 'medicines', 'details'}
        stoplist |= additional_stopwords
        sentences = [self.get_re().sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', s) for s in text.split('.') if s.strip()]
        sentences = [' '.join([word for word in s.split() if word.lower() not in stoplist])
                     for s in sentences]
        preprocessed_text = '. '.join(sentences)
        extractor.load_document(input=preprocessed_text)
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=40)
        for key in keyphrases:
            out.append(key[0])
        return out

    def get_distractors_wordnet(self, syn, word):
        distractors = []
        orig_word = word.lower().replace(" ", "_") if len(word.split()) > 0 else word.lower()
        hypernym = syn.hypernyms()
        if hypernym:
            for item in hypernym[0].hyponyms():
                name = item.lemmas()[0].name().replace("_", " ")
                if name != orig_word:
                    distractors.append(name.capitalize())
        return distractors

    def get_wordsense(self, sent, word):
        word = word.lower().replace(" ", "_") if len(word.split()) > 0 else word.lower()
        synsets = self.get_wn().synsets(word, 'n')
        if synsets:
            wup = self.get_max_similarity()(sent, word, 'wup', pos='n')
            adapted_lesk_output = self.get_adapted_lesk()(sent, word, pos='n')
            lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
            return synsets[lowest_index]
        else:
            return None

    def get_distractors_conceptnet(self, word):
        word = word.lower().replace(" ", "_") if len(word.split()) > 0 else word.lower()
        original_word = word
        distractor_list = []
        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (word, word)
        obj = self.get_requests().get(url).json()
        for edge in obj['edges']:
            link = edge['end']['term']
            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
            obj2 = self.get_requests().get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                if word2 not in distractor_list and original_word.lower() not in word2.lower():
                    distractor_list.append(word2)
        return distractor_list

    def get_distractors(self, input_file, keyword_sentence_mapping):
        key_distractor_list = {}
        for keyword in keyword_sentence_mapping:
            sentences = keyword_sentence_mapping[keyword]
            if sentences:
                csv_distractors = self.get_distractors_from_csv(input_file, keyword)
                if csv_distractors:
                    key_distractor_list[keyword] = csv_distractors
                else:
                    wordsense = self.get_wordsense(sentences[0], keyword)
                    if wordsense:
                        distractors = self.get_distractors_wordnet(wordsense, keyword)
                        if not distractors:
                            distractors = self.get_distractors_conceptnet(keyword)
                        if distractors:
                            key_distractor_list[keyword] = distractors
        return key_distractor_list

    def get_distractors_from_csv(self, input_file, keyword):
        encodings = ['utf-8', 'latin-1', 'utf-16']
        distractors_found = set()
        for encoding in encodings:
            try:
                with self.get_open(input_file, 'r', newline='', encoding=encoding) as csvfile:
                    reader = self.get_csv().DictReader(csvfile)
                    for row in reader:
                        key_concept = row['Key Concept']
                        distractors = row['Distractors'].split(', ')
                        if keyword.lower() in key_concept.lower():
                            distractors_found.update([distractor for distractor in distractors if distractor != keyword])
                        if keyword.lower() in distractors:
                            distractors_found.update([distractor for distractor in distractors if distractor != keyword])
                    if distractors_found:
                        return list(distractors_found)
                    break
            except UnicodeDecodeError:
                print(f"Error decoding file with encoding {encoding}. Trying another encoding...")
        return []

    def generate_mcqs(self, text_data):
        # Summarize the full text
        summarized_text = self.summarize_text(text_data)

        # Tokenize the summarized text into sentences
        sentences = self.tokenize_sentences(summarized_text)

        # Extract keywords from the full text
        keywords = self.get_nouns_multipartite(text_data)

        # Get sentences containing each keyword
        keyword_sentence_mapping = self.get_sentences_for_keyword(keywords, sentences)

        # Define the input file for distractors
        input_file = 'JAVA.csv'

        # Get distractors for each keyword
        key_distractor_list = self.get_distractors(input_file, keyword_sentence_mapping)

        # Generate MCQs from the collected data
        return self.generate_mcqs_from_data(keyword_sentence_mapping, key_distractor_list)

    def generate_mcqs_from_data(self, keyword_sentence_mapping, key_distractor_list):
        mcqs = []
        option_choices = ['a', 'b', 'c', 'd']

        for keyword in key_distractor_list:
            sentence = keyword_sentence_mapping[keyword][0]
            pattern = self.get_re().compile(keyword, self.get_re().IGNORECASE)
            output = pattern.sub(" _______ ", sentence)

            # Check if the number of distractors is less than 3
            if len(key_distractor_list[keyword]) < 3:
                # Include all distractors
                distractors = key_distractor_list[keyword]
                # Add padding to make up the required number of distractors
                distractors += [''] * (3 - len(distractors))
            else:
                # Selecting three random distractors
                distractors = self.get_random().sample(key_distractor_list[keyword], 3)

            # Adding the correct answer to the distractors
            distractors.append(keyword)

            # Shuffling the distractors
            self.get_random().shuffle(distractors)

            # Creating the MCQ dictionary
            mcq = {"question": output, "answer": keyword, "options": dict(zip(option_choices, distractors))}
            mcqs.append(mcq)

        return mcqs
