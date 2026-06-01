from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk

model_name = "nlp-waseda/roberta-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

dataset = load_from_disk("./dataset")

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

encoded = dataset.map(preprocess, batched=True)
encoded = encoded.rename_column("label", "labels")
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded,
)

trainer.train()
trainer.save_model("./model")