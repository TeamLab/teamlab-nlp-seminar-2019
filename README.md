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
|  |  [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) |  |MT  |
|  | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) |  |MT  |
|  | [Attention Is All You Need](https://paperswithcode.com/paper/attention-is-all-you-need) |  |  | LM |
|  | [Convolutional Sequence to Sequence Learning]() |  |  | MT |
|  | [Multimodal Machine Translation with Embedding Prediction]() |  |  | MT |
|  | ELMO |  |  | LM |
|  | BERT |  |  | LM |
|  | RoBERTa: A Robustly Optimized BERT Pretraining Approach |  |  | LM |
|  | OpenKE: An Open Toolkit for Knowledge Embedding |  |  | QA |
|  | Reading Wikipedia to Answer Open-Domain Questions |  |  | QA |
|  | Bidirectional Attention Flow for Machine Comprehension |  |  | QA |
|  | Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases |  |  | QA |
|  | RMDL: Random Multimodel Deep Learning for Classification |  |  | Classification |
|  | MaskGAN: Better Text Generation via Filling in the______ |  |  | Generation |
|  | Long Text Generation via Adversarial Training with Leaked Information |  | Generation | Generation |
|  | Deep Graph Convolutional Encoders for Structured Data to Text Generation | | Generation |
|  | GPT-2 |  |  | LM + Generation |
|  | XLNet |  |  | LM + Generation|
|  | Semi-Supervised Sequence Modeling with Cross-View Training | | NER |
|  | SciBERT: Pretrained Contextualized Embeddings for Scientific Text | | RE |
|  | BERT for Coreference Resolution: Baselines and Analysis | | CR |
|  | Sense Vocabulary Compression through the Semantic Knowledge of WordNet for Neural Word Sense Disambiguation | | WSA |
|  | CRIM at SemEval-2018 Task 9: A Hybrid Approach to Hypernym Discovery | | Hypernym |














## References
- [Ruder's blog](http://ruder.io/)
- [Tracking Progress in Natural Language Processing](https://github.com/sebastianruder/NLP-progress), by Sebastian Ruder
- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P02-1040). In Proceedings of ACL 2002. (Citation: 8,507)
- https://paperswithcode.com/task/joint-entity-and-relation-extraction
- [A Survey on Deep Learning for Named Entity Recognition](https://www.researchgate.net/publication/329946134_A_Survey_on_Deep_Learning_for_Named_Entity_Recognition)
