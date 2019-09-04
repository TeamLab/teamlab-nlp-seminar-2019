# Deep NLP seminar 2019

## 개요
TEAMLAB의 Deep NLP 세미나 발표자료를 보관하기 위한 저장소 입니다.
본 세미나는 딥러닝을 활용하여 NLP의 다양한 태스크들을 해결한 문제들에 대해 발표하고 구현하는 것을 목표로 합니다.

## 발표 및 코드 유의사항
- 구현에 한해서는 반드시 구현을 포함합니다.
- 영문 대표 데이터셋과 함께 반드시 한글 데이터셋에 대한 실험을 진행한다.
- 3개 이상의 데이터셋에 대하여, 실험을 실시하며 데이터 종류에 따른 전처리 코드를 포함한다.
- 최종산출물 코드는 반드시 `py` 파일로 작성하되, 실험결과 CSV 형태로 저장하는 자동화된 코드로 생성한다.
- 코드의 작성은 `pytorch`로 하되, `pytorch`의 예제 코드 작성방법에 준하여 코드를 작성한다.
- 가능한한 `zero-base`로 작성하되, `pytorch-transformer`, `spaCy`, `fast-ai` 등의 외부코드를 활용할 수 있다면 추가로 작성한다.

## 일정 및 주제
#### NLP trends
http://ruder.io/state-of-transfer-learning-in-nlp/

| Date | Paper | ppt | code | category |
|--|--|--|--| -- |
|  |  [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) |  |  | MT |
|  | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) |  |  | MT |
|  | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v5.pdf) |  |  | LM |
|  | [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf) |  |  | MT |
|  | [Multimodal Machine Translation with Embedding Prediction](https://www.aclweb.org/anthology/N19-3012) |  |  | MT |
|  | [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf) |  |  | LM |
|  | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) |  |  | LM |
|  | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf) |  |  | LM |
|  | [OpenKE: An Open Toolkit for Knowledge Embedding](https://www.aclweb.org/anthology/D18-2024) |  |  | QA |
|  | [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/pdf/1704.00051.pdf) |  |  | QA |
|  | [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/pdf/1611.01603) |  |  | QA |
|  | [Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases](https://arxiv.org/pdf/1903.02188) |  |  | QA |
|  | [RMDL: Random Multimodel Deep Learning for Classification](https://arxiv.org/pdf/1805.01890) |  |  | Classification |
|  | [MaskGAN: Better Text Generation via Filling in the___](https://arxiv.org/pdf/1801.07736) |  |  | Generation |
|  | [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/pdf/1709.08624) |  |  | Generation |
|  | [Deep Graph Convolutional Encoders for Structured Data to Text Generation](https://arxiv.org/pdf/1810.09995) | |  | Generation |
|  | [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) |  |  | LM + Generation |
|  | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237) |  |  | LM + Generation|
|  | [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/pdf/1809.08370) |  |  |NER |
|  | [SciBERT: Pretrained Contextualized Embeddings for Scientific Text](https://arxiv.org/pdf/1903.10676) |  |  | RE |
|  | [BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/pdf/1908.09091) |  |  | CR |
|  | [Sense Vocabulary Compression through the Semantic Knowledge of WordNet for Neural Word Sense Disambiguation](https://arxiv.org/pdf/1905.05677) |  |  | WSA |
|  | [CRIM at SemEval-2018 Task 9: A Hybrid Approach to Hypernym Discovery](https://www.aclweb.org/anthology/S18-1116) |  |  | Hypernym |














## References
- [Ruder's blog](http://ruder.io/)
- [Tracking Progress in Natural Language Processing](https://github.com/sebastianruder/NLP-progress), by Sebastian Ruder
- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P02-1040). In Proceedings of ACL 2002. (Citation: 8,507)
- https://paperswithcode.com/task/joint-entity-and-relation-extraction
- [A Survey on Deep Learning for Named Entity Recognition](https://www.researchgate.net/publication/329946134_A_Survey_on_Deep_Learning_for_Named_Entity_Recognition)
