import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, args):
        super(TransformerModel, self).__init__()
        self.config = BertConfig(
            vocab_size=vocab_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            hidden_size=args.hidden_size,
            position_embedding_type=args.position_embedding_type,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            output_hidden_states=args.output_hidden_states,
            use_cache=args.use_cache,
        )

        self._backbone = BertModel(config=self.config, add_pooling_layer=False)#, use_mlp=args.use_mlp)
        self.read_out = nn.Linear(self.config.hidden_size, 1)

    def forward(self, X, attention_mask, patch_states=None, extract_activation=None, head_mask=None):
        # hidden state of last layer
        embedding = self._backbone(
            input_ids=X,
            patch_states=patch_states,
            extract_activation=extract_activation,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask
        ).last_hidden_state
        output = self.read_out(embedding)

        return output
