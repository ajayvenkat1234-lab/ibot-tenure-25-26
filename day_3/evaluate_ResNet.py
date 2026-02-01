import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# No augmentation for test
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485 , 0.456 , 0.406],
                         [0.229 , 0.224 , 0.225])
])

# Load dataset
test_dataset = datasets.ImageFolder("data/test",transform=test_transforms)

# Create data loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Test samples:{len(test_dataset)}")

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load best model
model.load_state_dict(torch.load("best_model_ResNet.pth"))
model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []
correct_examples = {0 : [], 1 : []}
incorrect_examples = {0 : [], 1 : []}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = predicted[i].item()

            # Correct predictions (5 cats and 5 dogs)
            if true_label == pred_label:
                if len(correct_examples[true_label]) < 5:
                    correct_examples[true_label].append(
                        (images[i], labels[i], predicted[i])
                    )

            # Incorrect predictions (5 cats and 5 dogs)
            else:
                if len(incorrect_examples[true_label]) < 5:
                    incorrect_examples[true_label].append(
                        (images[i], labels[i], predicted[i])
                    )

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Show examples
def show_examples(examples, title):
    plt.figure(figsize=(20, 4))
    for i, (img, true, pred) in enumerate(examples):
        plt.subplot(1, 10, i + 1)
        img = img.cpu().permute(1, 2, 0)
        plt.imshow(img)
        plt.title(f"T:{true.item()} P:{pred.item()}")
        plt.axis("off")
    plt.suptitle(title)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

final_correct = correct_examples[0] + correct_examples[1]
final_incorrect = incorrect_examples[0] + incorrect_examples[1]

show_examples(final_correct, "Correct Predictions")
show_examples(final_incorrect, "Incorrect Predictions")

