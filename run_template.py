import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb


logger = logging.getLogger("__name__")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from;
    and addtional training arguments.
    """

    model_name_or_path: str = field(
        default="t5-small",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    tasks: str = field(
        default="question-answering",
        metadata={"help": "Task to fine-tune the model on"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    save_onnx: bool = field(default=False, metadata={"help": "Whether to save model to onnx."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})
    save_adapter: bool = field(
        default=False, metadata={"help": "Whether to export LoRA adapter before finish training."}
    )
    load_in_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Load model in LLM.int8 mode. Not recommended for model smaller than 13B."},
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Where to place the model weight blocks"},
    )
    use_early_stopping: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use early stop. The metric is set in --metric-for-best-model."},
    )
    early_stopping_patience: Optional[int] = field(
        default=3,
        metadata={"help": "The number of times to wait for the metric to improve before early stop."},
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={"help": "The threshold to measure the improvement of the metric."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "Use this if you want to use datasets.load_from_disk"},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    n_context: int = field(
        default=5,
        metadata={"help": "The number of documents to use for each question."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.data_dir is None
            and self.train_file is None
            and self.eval_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a data directory or a train/val/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json or a jsonl file."
            if self.eval_file is not None:
                extension = self.eval_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "jsonl",
                ], "`eval_file` should be a csv or a json or a jsonl file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`test_file` should be a csv or a json or a jsonl file."


def main():
    # Please store your API keys in a .env file in the root directory of this repository
    load_dotenv()

    # 1. Parse input arguments
    # Read more about TraniningArguments at
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[
            logging.FileHandler(filename="run.log", mode="a"),
            logging.StreamHandler(),
        ],
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # 3. Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        load_in_8bit=model_args.load_in_8bit,
        device_map=model_args.device_map if not training_args.deepspeed else None,
    )

    # 3.1 Prepare LoRA model
    # Read more at https://huggingface.co/docs/peft/quicktour
    if model_args.use_lora:
        peft_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        peft_model_name_or_path = (
            f"{model_args.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_")
        )
        training_args.run_name = peft_model_name_or_path + "_" + time.strftime("%y%m%d%H%M%S")
        if not training_args.hub_model_id:
            training_args.hub_model_id = peft_model_name_or_path

        model = get_peft_model(model, peft_config)
        logger.info(model.print_trainable_parameters())

    # 4. Prepare the datasets and data collator
    # You can either provide your own CSV/JSON/JSONL training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # Read more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    elif data_args.data_dir is not None:
        raw_datasets = load_from_disk(data_args.data_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.eval_file is not None:
            data_files["validation"] = data_args.eval_file
            extension = data_args.eval_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        if extension == "jsonl":
            extension = "json"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names

    if "question" in column_names:
        question_column_name = "question"
    else:
        raise (KeyError("Question column must be named 'question'"))

    if "answers" in column_names:
        answer_column_name = "answers"
    else:
        raise (KeyError("Answer column must be named 'answers'"))

    if "contexts" in column_names:
        context_column_name = "contexts"
    elif "ctxs" in column_names:
        context_column_name = "ctxs"
    elif "documents" in column_names:
        context_column_name = "documents"
    else:
        raise (KeyError("Document column must be named 'contexts', 'ctxs', or 'documents'"))

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # 4.2 Preprocess function
    # Modify this function according to your task
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples[question_column_name],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        labels = tokenizer(
            [answers[0] for answers in examples[answer_column_name]],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select the following argumentsd samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict need a test split")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select the following argumentsd samples again
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # 4.2 Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # 5. Metrics
    # Read more at https://huggingface.co/docs/evaluate
    metric = evaluate.load("rouge")

    def postprocess_function(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_function(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {
            "eval_rouge1": result["rouge1"] * 100,
            "eval_rouge2": result["rouge2"] * 100,
            "eval_rougeL": result["rougeL"] * 100,
            "eval_rougeLsum": result["rougeLsum"] * 100,
        }
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["eval_gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # 5. Set up Trainer
    # Read more at https://huggingface.co/docs/transformers/main_classes/trainer
    logger.info("Training/evaluation parameters %s", training_args)

    if training_args.push_to_hub:
        if training_args.hub_token is not None:
            pass
        elif os.getenv("HUGGING_FACE_HUB_TOKEN") is not None:
            training_args.hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        else:
            raise ValueError(
                "`--push_to_hub` require one of the followings: `--hub_token`, `huggingface-cli login`, or `HUGGING_FACE_HUB_TOKEN`."
            )

        if not training_args.hub_model_id:
            training_args.hub_model_id = model_args.model_name_or_path

    if training_args.report_to in ["all", "wandb"]:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    # Callbacks
    callbacks = []
    if model_args.use_early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping_patience,
            early_stopping_threshold=model_args.early_stopping_threshold,
        )
        callbacks.append(early_stopping_callback)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 6. Training
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint
        if checkpoint == "last-checkpoint":
            last_checkpoint = os.path.join(training_args.output_dir, checkpoint)
            if os.path.exists(last_checkpoint):
                checkpoint = last_checkpoint
            else:
                checkpoint = get_last_checkpoint(training_args.output_dir)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if model_args.use_lora:
            trainer.model.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 6.1 Save LoRA Adapter
    # Please set --do_train=0 --save_adapter=1 if you want to save adapter halfway
    if model_args.use_lora and model_args.save_adapter:
        model.load_state_dict(torch.load(training_args.output_dir + "/pytorch_model.bin"))
        model.save_pretrained(training_args.output_dir)
        logger.info("Successfully saved adapter to %s", training_args.output_dir)

    # 7. Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 8. Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, metric_key_prefix="test")
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    # 9. Create model card and push to hub
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": model_args.tasks}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    trainer.create_model_card(**kwargs)

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
