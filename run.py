import os
import yaml
import argparse
from medval.pipeline import MedVAL
from tqdm import tqdm
import dspy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Name of the YAML file under the directory configs/")
    args = parser.parse_args()
    args.config = f"configs/{args.config}.yaml" if not args.config.endswith(".yaml") else args.config
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get("API_KEY") if config["api_key"] == "${API_KEY}" else config["api_key"]
    n_samples = int(config["n_samples"]) if config["n_samples"] is not None else None
    threshold = config.get("threshold", None)
    input_csv = config.get("input_csv", None)

    if (config["data"] == "train") and (config["model"].startswith("local")):
        medval = MedVAL(config["tasks"], "local/" + config["local_model_path"], config["api_base"], api_key, config["data"], n_samples, config["debug"], config["method"], threshold, input_csv)
    else:
        medval = MedVAL(config["tasks"], config["model"], config["api_base"], api_key, config["data"], n_samples, config["debug"], config["method"], threshold, input_csv)

    if config["data"] == "train":
        medval.generator._compiled = True
        trainset, valset = medval.load_data()
        agents_path = f"agents/"
        output_dir = os.path.join(agents_path, f"{config['student_model'].split('/')[-1]}/")
        os.environ["DSPY_FINETUNEDIR"] = output_dir
        if (config["data"] == "train") and (config["model"].startswith("local")):
            medval_student = medval
        else:
            medval_student = MedVAL(config["tasks"], config["model"], config["api_base"], api_key, config["data"], n_samples, config["debug"], config["method"], threshold, input_csv)
            medval_student.student_model = "local/" + config["local_model_path"]
            medval_student._configure_lm()
            medval_student.generator._compiled = True
        
        if (config['student_model'].startswith("openai")) or (config['student_model'].startswith("azure")):
            optimizer = dspy.BootstrapFinetune(metric=medval.validator_metric, num_threads=config["num_threads"])
        else:
            optimizer = dspy.BootstrapFinetune(metric=medval.validator_metric, num_threads=config["num_threads"], train_kwargs=dict(num_train_epochs=config["num_epochs"], use_peft=True, use_quantization=True, output_dir=output_dir))
        optimized_validator = optimizer.compile(medval_student, teacher=medval, trainset=trainset)
        
    elif config["data"] == "test":
        medval.generator._compiled = True
        if config['model'].startswith("local"):
            agents_path = f"agents/"
            model_path = os.path.join(agents_path, f"{config['model'].split('/')[-1]}")
            medval.student_model = "local/" + config["local_model_path"]
            medval._configure_lm()
            medval.get_lm().launch()
        
        df, dataset = medval.load_data()

        for i, x in enumerate(tqdm(dataset)):
            try:
                result = medval.forward(reference=x["reference"], task=x["task"], candidate=x["candidate"])
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                result = {"attack_prediction": None, "err": None, "reason": None}
            
            if config["debug"]:
                print(dspy.inspect_history(n=2))
                exit()
            
            df.at[i, "lm_risk_grade"] = result["attack_prediction"]
            df.at[i, "lm_error_assessment"] = result["err"]
            df.at[i, "lm_reasoning"] = result["reason"]
        
        df = df[["#", "id", "task", "reference", "target", "candidate", "lm_reasoning", "lm_error_assessment", "physician_error_assessment", "lm_risk_grade", "physician_risk_grade"]]
        df = df.rename(columns={"reference": "input", "target": "reference_output", "candidate": "output"})
        medval.save_results(df, config["method"])
        
        if config['model'].startswith("local"):
            medval.get_lm().kill()