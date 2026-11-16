
import torch
import torch.nn as nn
from model import ConvModel
from dataset import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ConvModel().to(device)

num_epochs = 100

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.0001
)

train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

best_val_accuracy = 0.0
best_epoch = 0


for epoch in range(num_epochs):

    # TRAINING
    model.train()

    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = 100 * train_correct / train_total


    # VALIDATION
    model.eval()

    val_loss_sum = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            val_loss_sum += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = 100 * val_correct / val_total


    # BEST MODEL CHECKPOINT
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(">>> NEW BEST MODEL SAVED!")


    # LEARNING RATE SCHEDULER
    scheduler.step()


    # EPOCH RESULTS
    print(f"Epoch: {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.3f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.3f}")
    print(f"Val Accuracy: {val_accuracy:.2f}%")
    print("-" * 35)


print("TRAINING COMPLETED")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
