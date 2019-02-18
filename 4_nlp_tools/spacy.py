import spacy

# python -m spacy download en
import en_core_web_sm
nlp = en_core_web_sm.load()

# Load English tokenizer, tagger, parser, NER and word vectors
# nlp = spacy.load('en')
# nlp = spacy.load('en_core_web_sm')

# Process whole documents
text = (u"When Sebastian Thrun started working on self-driving cars at "
        u"Google in 2007, few people outside of the company took him "
        u"seriously. “I can tell you very senior CEOs of major American "
        u"car companies would shake my hand and turn away because I wasn’t "
        u"worth talking to,” said Thrun, now the co-founder and CEO of "
        u"online higher education startup Udacity, in an interview with "
        u"Recode earlier this week.")
doc = nlp(text)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

# Determine semantic similarities
doc1 = nlp(u"my fries were super gross")
doc2 = nlp(u"such disgusting fries")
similarity = doc1.similarity(doc2)
print(doc1.text, doc2.text, similarity)

doc1 = nlp(u"you are so great")
doc2 = nlp(u"you are not bad at all")
doc3 = nlp(u"you are not so great")
similarity = doc1.similarity(doc2)
print(doc1.text, doc2.text, similarity)
similarity = doc1.similarity(doc3)
print(doc1.text, doc3.text, similarity)


# https://spacy.io/usage/spacy-101
for token in doc1:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)