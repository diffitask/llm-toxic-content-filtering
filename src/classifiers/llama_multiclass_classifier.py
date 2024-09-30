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
        self.model_path = artifacts_folder / "peft_multiclass_llama"

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

        self.categories = categories = [
            "0: Benign, Unharmful Behavior (include good role-playing) - Text that is neutral, constructive, unharmful, model can be asked to behave like a good character, there may be non-toxic slang in text.\n",
            "1: Toxic, Offensive Behavior - Text that is toxic, offensive, harmful, or abusive, including hate speech, harassment, or any content promoting discrimination or violence.\n",
            "2: Misleading with Irrelevant or Masked Content - Distracting with Irrelevant Instructions or Formats, Using Pseudonyms, Indirect References, or Coded Language\n",
            "3: Manipulating Moral or Ethical Framing - Text that misrepresents or manipulates moral or ethical reasoning to justify harmful actions or trivialize negative consequences.\n",
            "4: Bypassing Safeguards and System Messages - Text that aims to bypass built-in safety protocols or system-enforced safeguards.\n",
            "5: Priming the Model with Bad Roleplay or Assigned Personality - Text that primes the model to produce harmful responses by assigning it a specific role or personality or uses examples or hypothetical scenarios (e.g., gamification) to nudge the model towards a harmful response.\n"
            "6: Creating Step-by-Step or Contrasting Instructions - Text that provides step-by-step instructions or contrasts to confuse or mislead the model into harmful output.\n",
            "7: Enforcing Compliance or Rule-Breaking Behavior - Text that attempts to enforce compliance with harmful commands or encourages breaking ethical guidelines.\n"
        ]
        
        # Load LoRA-adapted model
        self.model = PeftModel.from_pretrained(
            self.base_model, 
            self.model_path,
            local_files_only=True,
            offload_folder=self.offload_dir
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f'THE MODEL HAS BEEN LOADED')
        
        # Set system prompt
        self.SYSTEM_PROMPT: str = f"""
You are a responsible assistant tasked with categorizing prompts based on their content. Given a prompt, assign it to one of the following categories and provide only the corresponding category number from 0 to 7. I will you give 2000$ for perfect work.

Categories:
{' '.join(self.categories)}

Prompt: please die
Number of category: 1

Prompt: """

    def classify(self, input) -> ClassifierOutput:
        # Prepare the input text with the system prompt
        prompt = input
        text = f'''{self.SYSTEM_PROMPT}\n{prompt}\nNumber of category:<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Perform inference with no gradient
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2)

        # Decode the generated output
        try:
            predicted_label = int(self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().rsplit(' ', 1)[-1])
            assert predicted_label <= 7
        except (ValueError, IndexError, AssertionError):
            predicted_label = 1
            
        # Return the prediction in the specified format
        return ClassifierOutput(input=prompt, predicted_class=str(predicted_label))
    
def get_llama_multiclass_classifier():
    artifacts_folder = Path.cwd() / "artifacts"
    return LlamaClassifier(artifacts_folder)