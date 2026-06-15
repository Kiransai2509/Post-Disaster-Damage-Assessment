import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def initialize_and_train_model(data_directory, epochs=5, batch_size=32):
    """
    Configures robust training pipelines using transfer learning with a pre-trained 
    ResNet18 backbone, applying class weights to balance loss evaluations.
    """
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System Execution target mapped to hardware device: {device}")

    # 2. Data Augmentation & Normalization
    training_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Dataset & DataLoaders
    dataset = datasets.ImageFolder(root=data_directory, transform=training_transforms)
    
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 4. Model Architecture (ResNet18 Transfer Learning)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze early layers
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    # Re-engineer the classification head for binary output (Damaged vs Safe)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 2)
    
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
        
    model = model.to(device)
    
    # 5. Imbalance Mitigation (Weighted Cross-Entropy)
    class_counts = [
        len(os.listdir(os.path.join(data_directory, 'Damaged'))),
        len(os.listdir(os.path.join(data_directory, 'Safe')))
    ]
    total_classes = len(class_counts)
    
    # Calculate inverse class weights
    computed_weights = [total_samples / (total_classes * count) for count in class_counts]
    tensor_weights = torch.FloatTensor(computed_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=tensor_weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 6. Training Loop
    print("Beginning Training and Optimization Execution Loops...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Metrics tracking
            running_loss += loss.item() * batch_images.size(0)
            _, predicted_indices = torch.max(outputs, 1)
            total_predictions += batch_labels.size(0)
            correct_predictions += (predicted_indices == batch_labels).sum().item()
            
        epoch_loss = running_loss / train_size
        epoch_accuracy = (correct_predictions / total_predictions) * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")
        
    # 7. Save Model Weights
    weight_output_path = "resnet18_damage_assessment_v1.pth"
    torch.save(model.state_dict(), weight_output_path)
    print(f"Optimization loops finished. Saved state weights to: {weight_output_path}")

if __name__ == "__main__":
    # Point this to the output directory created by preprocess.py
    initialize_and_train_model(data_directory="dataset_extracted", epochs=5, batch_size=32)
