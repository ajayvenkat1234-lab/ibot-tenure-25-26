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

# Load pretrained MobileNetV2
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)


# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load best model
model.load_state_dict(torch.load("best_model_MobileNetV2.pth"))
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
print(f'Test Accuracy_MobileNetV2: {test_accuracy:.2f}%')

