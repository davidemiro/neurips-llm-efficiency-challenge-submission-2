import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutput

class AutoModelForCausalLMWithResidualDropout(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.start_layer_dropout = torch.nn.Dropout(config.start_layer_dropout)
        self.end_layer_dropout = torch.nn.Dropout(config.end_layer_dropout)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask)

        # Apply residual dropout to the decoder outputs
        num_layers = len(outputs)

        for i in range(num_layers):
            # gradually increase the dropout probability following the LIMA paper
            current_dropout_probability = self.start_layer_dropout + (
                    self.end_layer_dropout - self.start_layer_dropout) * i / num_layers
            outputs[i] = BaseModelOutput.residual(outputs[i], current_dropout_probability)

        # Compute the loss
        loss = self.compute_loss(outputs, labels)

        return loss, outputs