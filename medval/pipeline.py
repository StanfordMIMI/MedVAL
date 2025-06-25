import dspy
import os
import json
import pandas as pd
import random
from dspy.datasets import DataLoader
from utils.prompts import adversarial_attacks, adversarial_attack_base, error_categories
from medval.generator import MedVAL_Generator
from medval.validator import MedVAL_Validator
from dspy.clients.lm_local import LocalProvider
from datasets import load_dataset

def scale_to_unit_interval(val, num_levels):
    return (val - 1) / (num_levels - 1)

class MedVAL(dspy.Module):
    def __init__(self, tasks, model, api_base, api_key, data, n_samples, debug, method, threshold):
        self.tasks = tasks
        self.model_name = model
        self.api_base = api_base
        self.api_key = api_key
        self.data = data
        self.n_samples = n_samples
        self.debug = debug
        self.method = method
        self.threshold = threshold
        self.student_model = None
        self.generator = dspy.ChainOfThought(MedVAL_Generator).deepcopy()
        self.validator = dspy.ChainOfThought(MedVAL_Validator).deepcopy()
        self.prompts = self._load_prompts()
        self._configure_lm()
        self.dl = DataLoader()

    def _load_prompts(self):
        with open("utils/task_prompts.json", 'r') as file:
            return json.load(file)

    def _configure_lm(self):
        if (self.data == "train") and (self.model_name.startswith("local")):
            dspy.settings.experimental = True
            lm = dspy.LM(model=f"openai/local:{"/".join(self.model_name.split("/")[1:])}", provider=LocalProvider())
            lm.launch()
            dspy.configure(lm=lm)
            
        else:
            lm = dspy.LM(model=self.model_name, api_base=self.api_base, api_key=self.api_key)
            if self.student_model != None:
                dspy.settings.experimental = True
                self.generator.set_lm(lm)
                if not self.student_model.startswith("local"):
                    self.validator.set_lm(dspy.LM(model=self.student_model, api_base=self.api_base, api_key=self.api_key))
                else:
                    self.student_model = "/".join(self.student_model.split("/")[1:])
                    self.validator.set_lm(dspy.LM(model=f"openai/local:{self.student_model}", provider=LocalProvider()))
            else:
                dspy.configure(lm=lm)

    def load_data(self):
        hf_dataset = load_dataset("stanfordmimi/MedVAL-Bench")
        dataset_split = hf_dataset["train"] if self.data == "train" else hf_dataset["test"]
        df = dataset_split.to_pandas()
        df = df[df['task'].isin(self.tasks)]
        df = df.head(self.n_samples) if self.n_samples is not None else df
        print(f"\nTasks included: {', '.join(self.tasks)}")
        print(f"\nTotal # of samples: {len(df)}\n\n")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) if self.data == "train" else df.reset_index(drop=True)
        temp_csv_path = f"temp.csv"
        df.to_csv(temp_csv_path, index=False)
            
        if self.data == "train":
            full_dataset = self.dl.from_csv(temp_csv_path, fields=("input", "reference_output", "task"), input_keys=("input", "reference_output", "task"))
            os.remove(temp_csv_path)
            return full_dataset, None
        else:
            full_dataset = self.dl.from_csv(temp_csv_path, fields=("input", "reference_output", "task", "output"), input_keys=("input", "task", "output"))
            os.remove(temp_csv_path)
            return df, full_dataset

    def generate(self, input, attack_level, task):
        adversarial_instruction = self.prompts[task] + adversarial_attack_base + adversarial_attacks[attack_level-1] + "\n" + error_categories
        result = self.generator(instruction=adversarial_instruction, input=input)
        return result["output"]

    def forward(self, input, task, output=None, reference_output=None):
        if output == None:
            random.seed(hash(input) % (2**32))
            attack_level = random.randint(1, len(adversarial_attacks))
            output = self.generate(input=input, attack_level=attack_level, task=task)

        result = self.validator(instruction=self.prompts[task], input=input, output=output)
        
        if (self.data == "train"):
            output_clean = self.generate(input=input, attack_level=1, task=task) if reference_output == None else reference_output
            result_clean = self.validator(instruction=self.prompts[task], input=input, output=output_clean)
            return dspy.Prediction(reason=result["reasoning"], err=result["errors"], attack_prediction=result["risk_level"], attack_level=attack_level, clean_prediction=result_clean["risk_level"])
        
        return dspy.Prediction(reason=result["reasoning"], err=result["errors"], attack_prediction=result["risk_level"])
    
    def validator_metric(self, example, pred, trace=None):
        delta = scale_to_unit_interval(pred["attack_level"], num_levels=len(adversarial_attacks))
        pred_clean_score = scale_to_unit_interval(pred["clean_prediction"], num_levels=len(adversarial_attacks))
        pred_adv_score = scale_to_unit_interval(pred["attack_prediction"], num_levels=len(adversarial_attacks))

        absolute_consistency = (pred_clean_score ** 2) + (pred_adv_score - delta)**2
        relative_consistency = (pred_adv_score - pred_clean_score - delta)**2
        total_loss = absolute_consistency + relative_consistency
        metric_value = 1 - total_loss / 6

        if self.debug:
            print(dspy.inspect_history(n=4))
            exit()

        if (trace is not None) or (self.method == "finetune"): return metric_value >= self.threshold
        return metric_value
        
    def save_results(self, df, method=None):
        df = df.where(pd.notnull(df), "None")
        df["lm_error_assessment"] = df["lm_error_assessment"].str.replace("\n\n", "\n", regex=False).str.replace("\n \n", "\n", regex=False).str.replace("\\n", "\n", regex=False)

        results_path = f"results/{method}/"
        os.makedirs(results_path, exist_ok=True)
        file_path = f"{results_path}{self.model_name.split('/')[-1]}.csv"
        df.to_csv(file_path, index=False)
        print(f"\nResults saved to: {file_path}\n")