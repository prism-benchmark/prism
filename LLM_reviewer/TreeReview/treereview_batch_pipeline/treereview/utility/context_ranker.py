import logging
from typing import List
import torch
import tiktoken
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class ContextRanker:
    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 device_map: str = "cuda",
                 model_config: dict = None,
                 openai_api_config: dict = None,
                 condition_modifier=" We can get the answer to this question in the given documents."):
        self.model_name = model_name
        self.openai_api_config = openai_api_config or {}
        self._condition_modifier = condition_modifier
        self.oai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self._loaded_model_name = None
        self.device = (
            device_map
            if any(key in device_map for key in ["cuda", "cpu", "mps"])
            else "cuda"
        )
        self._load_model(self.model_name, model_config)

    def _load_model(
        self, model_name: str,
        model_config: dict = None
    ):
        model_config = model_config or {}
        logger.info(f"Loading model {model_name}")
        trust_remote_code = model_config.get("trust_remote_code", True)
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = trust_remote_code
        config = AutoConfig.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
        if model_config.get("pad_to_left", True):
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = (
                config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
            )
        MODEL_CLASS = (
            AutoModelForTokenClassification
            if any("ForTokenClassification" in ar for ar in config.architectures)
            else AutoModelForCausalLM
        )
        
        if "cuda" in self.device or "cpu" in self.device:
            model = MODEL_CLASS.from_pretrained(
                model_name,
                torch_dtype=model_config.pop(
                    "torch_dtype", "auto" if self.device == "cuda" else torch.float32
                ),
                device_map=self.device,
                config=config,
                ignore_mismatched_sizes=True,
                **model_config,
            )
        else:
            model = MODEL_CLASS.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=model_config.pop("torch_dtype", "auto"),
                pad_token_id=tokenizer.pad_token_id,
                **model_config,
            )
        self.tokenizer = tokenizer
        self.model = model
        self.context_idxs = []
        self.max_position_embeddings = config.max_position_embeddings

    def _get_ppl(
            self,
            text: str,
            granularity: str = "sentence",
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            return_kv=False,
            end=None,
            condition_mode: str = "none",
            condition_pos_id: int = 0,
    ):
        if input_ids is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].to(self.device)
            attention_mask = tokenized_text["attention_mask"].to(self.device)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)
        with torch.no_grad():
            response = self.model(
                input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values

        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1: end].contiguous()
        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        if condition_mode == "before":
            loss = loss[:condition_pos_id]
        elif condition_mode == "after":
            loss = loss[condition_pos_id:]
        res = loss.mean() if granularity == "sentence" else loss
        return (res, past_key_values) if return_kv else res

    def _get_token_length(
        self,
        text: str,
        add_special_tokens: bool = True,
        use_oai_tokenizer: bool = False,
    ):
        if use_oai_tokenizer:
            return len(self.oai_tokenizer.encode(text))
        else:
            return len(
                self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids
            )

    def _get_condition_ppl(
            self,
            text: str,
            question: str,
            condition_in_question: str = "none",
            granularity: str = "sentence",
    ):
        if condition_in_question == "none":
            return self._get_ppl(text, granularity=granularity)
        elif condition_in_question == "before":
            return self._get_ppl(
                question + text,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self._get_token_length(question) - 1,
            )
        elif condition_in_question == "after":
            return self._get_ppl(
                text + question,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self._get_token_length(text) - 1,
            )

    def _get_rank_results(
        self,
        context: list,
        question: str,
        condition_in_question: str,
        context_tokens_length: list,
    ):
        def get_distance_longllmlingua(corpus, query):
            context_ppl = [
                self._get_condition_ppl(
                    d,
                    query
                    + self._condition_modifier,
                    condition_in_question,
                )
                - dl * 2 / 250 * 0
                for d, dl in zip(corpus, context_tokens_length)
            ]
            sort_direct = -1 if condition_in_question == "none" else 1
            ys = sorted(enumerate(context_ppl), key=lambda x: sort_direct * x[1])
            return ys
        return get_distance_longllmlingua(context, question)

    def rank_chunks(self, chunks:List[str], question:str, top_k=5):
        context_tokens_length = [self._get_token_length(c) for c in chunks]
        ranks_with_score =  self._get_rank_results(
                                      chunks,
                                      question,
                                      condition_in_question="after",
                                      context_tokens_length=context_tokens_length)
        ranked_idxes = [int(idx[0]) for idx in ranks_with_score]
        return [chunks[i] for i in ranked_idxes[:top_k]]



