from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig
)
from transformers.pipelines import AggregationStrategy
import numpy as np
import pandas as pd
from typing import Dict
import torch
import nltk.data
nltk.download('punkt')


class ColBERTConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True

class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    """
    config_class = ColBERTConfig
    base_model_prefix = "bert_model"

    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)

        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor]):

        query_vecs = self.forward_representation(query)
        document_vecs = self.forward_representation(document)

        score = self.forward_aggregation(query_vecs, document_vecs, query["attention_mask"], document["attention_mask"])
        return score

    def forward_representation(self,
                               tokens,
                               sequence_type=None) -> torch.Tensor:

        vecs = self.bert_model(**tokens)[0]  # assuming a distilbert model here
        vecs = self.compressor(vecs)

        # if encoding only, zero-out the mask values so we can compress storage
        if sequence_type == "doc_encode" or sequence_type == "query_encode":
            vecs = vecs * tokens["tokens"]["mask"].unsqueeze(-1)

        return vecs

    def forward_aggregation(self, query_vecs, document_vecs, query_mask, document_mask):

        # create initial term-x-term scores (dot-product)
        score = torch.bmm(query_vecs, document_vecs.transpose(2, 1))

        # mask out padding on the doc dimension (mask by -1000, because max should not select those, setting it to 0 might select them)
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~exp_mask] = - 10000

        # max pooling over document dimension
        score = score.max(-1).values

        # mask out paddding query values
        score[~(query_mask.bool())] = 0

        # sum over query values
        score = score.sum(-1)

        return score


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])


def sentence_split(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    new_text = tokenizer.tokenize(text)
    return new_text


def score_passage_query(passages, queries, tokenizer, model):
    rank_dict = {}
    for query in queries:
        rank_dict[query] = {}
        for idx, passage in enumerate(passages):
            passage_input = tokenizer(passage, return_tensors="pt")
            query_input = tokenizer(query)
            query_input.input_ids += [103] * 8
            query_input.attention_mask += [1] * 8
            query_input["input_ids"] = torch.LongTensor(query_input.input_ids).unsqueeze(0)
            query_input["attention_mask"] = torch.LongTensor(query_input.attention_mask).unsqueeze(0)
            score_p = model.forward(query_input, passage_input).squeeze(0)
            rank_dict[query][idx] = float(score_p)
    return rank_dict


def retrieve_top_n_passage(scores, n_passages, ratio=0.3):
    top_passages = int(n_passages * ratio)
    retrieved = {}
    for query in scores:
        ranking = sorted(ranked_chapter_1[query].items(), key=lambda x: x[1], reverse=True)[:top_passages]
        # sort ranking again, to rearrange by appearance order for each passage in the corpus
        ranking.sort(key=lambda x: x[0])
        ranking = [passage_info[0] for passage_info in ranking]
        retrieved[query] = ranking
    return retrieved


if __name__ == "__main__":

    seed = ["data"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = ColBERT.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco")

    # load the data...
    # df = pd.read_csv("~/Documents/R&D/Suggestion Engine/data/shock_analysis.csv")
    df = pd.read_csv("~/Documents/R&D/Suggestion Engine/data/The digital transformation of the automotive industry.csv")
    # df = pd.read_csv("~/Documents/R&D/Suggestion Engine/data/it_emerging_markets.csv")
    text = df['Content'].values

    for i, page in enumerate(text):

        # select the n-th chapter
        sentence_list_chapter_1 = sentence_split(text[i])
        # print(len(sentence_list_chapter_1))

        # rank the passages using the information retrieval package...
        ranked_chapter_1 = score_passage_query(sentence_list_chapter_1, seed, tokenizer=tokenizer, model=model)
        # .. and get
        retrieval = retrieve_top_n_passage(ranked_chapter_1, n_passages=len(sentence_list_chapter_1))
        # print(retrieval)

        chapter_relevant = []
        for passage_nb in retrieval[seed[0]]:
            chapter_relevant.append(sentence_list_chapter_1[passage_nb])

        print("Top 5 relevant sentences: ", chapter_relevant[:5])
        # print(chapter_relevant[:3])
        chapter_relevant = ' '.join(chapter_relevant)

        model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
        extractor = KeyphraseExtractionPipeline(model=model_name)
        keyphrases_w_retrieval = extractor(chapter_relevant)
        print(f"Keywords with info retrieval: {keyphrases_w_retrieval}\n")

        # Load pipeline
        # model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
        # extractor = KeyphraseExtractionPipeline(model=model_name)
        #
        # df = pd.read_csv("~/Documents/R&D/Suggestion Engine/data/it_emerging_markets.csv")
        # text = df['Content'].values
        #

        keyphrases = extractor(text[0])
        print(f"Keywords without info retrieval: {keyphrases}\n")

        diff_values = list(set(keyphrases_w_retrieval) - set(keyphrases))  # call to this function
        print(diff_values, "\n")

