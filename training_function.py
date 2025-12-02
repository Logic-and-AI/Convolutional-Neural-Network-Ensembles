import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import time

def train_meta_model(meta_cnn, train_loader, test_loader, device, lr=8e-5, weight_decay=1e-3, max_epochs=1000, target_accuracy=94):
    optimizer = optim.AdamW(meta_cnn.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_accuracies = []
    test_accuracies = []

    start_time = time.time()

    for epoch in range(max_epochs):
        meta_cnn.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = meta_cnn(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        train_acc = 100 * correct_train / total_train
        train_accuracies.append(train_acc)

        meta_cnn.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = meta_cnn(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        if test_acc >= target_accuracy:
            print(" Target accuracy reached. Stopping training.")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nTotal Training Time: {int(minutes)} min {int(seconds)} sec")

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
