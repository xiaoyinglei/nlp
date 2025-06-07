from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import cross_entropy
import pandas as pd

# 自动选择设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. 加载数据集
dataset = load_dataset("emotion")

# 2. 加载 tokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# 3. 分词函数
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# 4. 编码数据集
encoded = dataset.map(tokenize, batched=True)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. 加载预训练模型（分类任务）
num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)

# 6. 指标函数
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# 7. 训练参数
batch_size = 64
logging_steps = len(encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    report_to="none",  # 关闭 wandb 或 tensorboard
    log_level="error"
)

# 8. Trainer 初始化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

# 9. 开始训练
trainer.train()

# 10. 预测评估
preds_output = trainer.predict(encoded["validation"])
print(preds_output.metrics)

# 11. 计算逐样本 loss 和预测标签
def forward_pass_with_label(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
    return {
        "loss": loss.cpu().numpy(),
        "predicted_label": pred_label.cpu().numpy()
    }

encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
encoded["validation"] = encoded["validation"].map(forward_pass_with_label, batched=True)

# 12. 转换为 pandas 便于分析
encoded.set_format("pandas")
label_int2str = dataset["train"].features["label"].int2str
df = encoded["validation"][:][["text", "label", "predicted_label", "loss"]]
df["label"] = df["label"].apply(label_int2str)
df["predicted_label"] = df["predicted_label"].apply(label_int2str)
print(df.head())