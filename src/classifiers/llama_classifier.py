from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.schemas import ClassifierOutput
from src.classifiers.classifier_interface import ClassifierInterface
import torch

class LlamaClassifier(ClassifierInterface):
    def __init__(self, artifacts_folder):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f'USING DEVICE: {self.device}')
        # self.device = torch.device('mps')
        
        # Initialize tokenizer and model
        self.tokenizer_path = artifacts_folder / "saiga_llama3_tokenizer"
        self.base_model_path = artifacts_folder / "saiga_llama3_base_model"
        self.model_path = artifacts_folder / "peft_llama"

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_path,
            local_files_only=True,
        )
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_eos_token = True
        self.tokenizer.add_bos_token = True
        self.offload_dir=self.model_path / 'offload'

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            offload_folder=self.offload_dir,
        )
        
        # Load LoRA-adapted model
        self.model = PeftModel.from_pretrained(
            self.base_model, 
            self.model_path,
            local_files_only=True,
            offload_folder=self.offload_dir,
            assign=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f'THE MODEL HAS BEEN LOADED')
        
        # Set system prompt
        self.SYSTEM_PROMPT: str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\
You're the responsible assistant. Examine the text carefully and determine whether it contains jailbreaks or is toxic. \
If the text is toxic or contains jailbreaks, print 1. Otherwise, print 0.\
\
<|eot_id|><|start_header_id|>user<|end_header_id|>\
Statement:"

    def classify(self, input) -> ClassifierOutput:
        # Prepare the input text with the system prompt
        prompt = input
        text = f'''{self.SYSTEM_PROMPT} {prompt}\nAnswer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Perform inference with no gradient
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2)

        # Decode the generated output
        try:
            predicted_label = int(self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()[-1])
            assert predicted_label <= 1
        except (ValueError, IndexError, AssertionError):
            predicted_label = 1  # Return 0 if conversion fails
            
        # Return the prediction in the specified format
        return ClassifierOutput(input=prompt, predicted_class=str(predicted_label))
    
def get_llama_classifier():
    artifacts_folder = Path.cwd() / "artifacts"
    return LlamaClassifier(artifacts_folder)