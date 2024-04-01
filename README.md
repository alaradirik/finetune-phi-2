## Fine-tuning Phi 2 for Persona-Grounded Chat
This project is a tutorial on parameter-efficient fine-tuning (PEFT) and quantization of the [Phi 2](https://huggingface.co/microsoft/phi-2/) model for persona-grounded chat. We use LoRA for PEFT and 4-bit quantization to compress the model, and fine-tune the model on a new persona-based chat dataset - [nazlicanto/persona-based-chat](https://huggingface.co/datasets/nazlicanto/persona-based-chat). Refer to the dataset page and Hugging Face model [page](https://huggingface.co/nazlicanto/phi-2-persona-chat) for additional details.

## Usage
Start by cloning the repository, setting up a conda environment and installing the dependencies. We tested our scripts with python 3.9 and CUDA 11.7.
```
git clone https://github.com/alaradirik/finetune-phi-2.git
cd finetune-phi-2

conda create -n llm python=3.9
conda activate llm
pip install -r requirements.txt
```

You can finetune the model on chat [dataset](https://huggingface.co/datasets/nazlicanto/persona-based-chat) or another dataset. Note that you will need to have the same features as our dataset and pass in your HF Hub token as an argument if using a private dataset. Fine-tuning takes about 9 hours on a single A40, you can either use the default accelerate settings or configure it to use multiple GPUs. To fine-tune the model:
```
accelerate config default

python finetune_phi.py --dataset=<HF_DATASET_ID_OR_PATH> --base_model="microsoft/phi-2" --model_name=<YOUR_MODEL_NAME> --auth_token=<HF_AUTH_TOKEN> --push_to_hub
```

One model training is completed, only the fine-tuned (LoRA) parameters are saved, which are loaded to overwrite the corresponding parameters of the base model during testing. To test the fine-tuned model with a random sample selected from the dataset, run `python test.py`.


## License
The code and trained model are licensed under the [MIT license](https://github.com/adirik/finetune-phi-2/blob/main/LICENSE).