import torch
import torch.nn as nn
import math

from transformers import AutoModelForCausalLM, AutoTokenizer


class LMScorer(nn.Module):
    def __init__(self, llm_path, lm_largest_texts_length=500):
        super().__init__()
        lm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()
        
        self.lm = lm.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

        self.lm_largest_texts_length = lm_largest_texts_length
    
    def compute_lm_scores(self, partial_texts):
        model_inputs = self.tokenizer(partial_texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
        
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask

        with torch.no_grad():
            outputs = self.lm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = outputs[0]
        logits = self.lm.lm_head(hidden_states)
        logits = logits.float()

        scores = []
        for logits, m, labels in zip(logits, attention_mask, input_ids):
            token_sum = m.sum()
            logp_total = logits[: token_sum - 1].log_softmax(dim=-1)

            logp = 0
            for i in range(token_sum - 1):
               logp += logp_total[i, labels[i + 1]]
            
            logp /= token_sum - 1 + 1e-9

            scores.append(logp)

        return scores

    def forward(self, texts):
        total_text_length = len(texts)
        total_score = []
        max_loop = int(math.ceil(total_text_length / self.lm_largest_texts_length))

        for i in range(max_loop):
            total_score.extend(
                self.compute_lm_scores(
                    texts[i * self.lm_largest_texts_length: min(total_text_length, (i + 1) * self.lm_largest_texts_length)]
                )
            )
        
        return total_score
