import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
import torch.utils.data
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast

if __name__ == '__main__':
    def main_SimpleNN():
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
                self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.fc1 = nn.Linear(128 * 3 * 3, 512)  # FashionMNIST veri kümesi 28x28 boyutunda olduğu için
                self.fc2 = nn.Linear(512, 10)  # FashionMNIST veri kümesi 10 sınıfa sahiptir

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.pool(F.relu(self.conv3(x)))
                x = x.view(-1, 128 * 3 * 3)  # Flatten işlemi
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)  # Log-olasılık değerleri elde etmek için softmax fonksiyonunu kullanın

        net = SimpleNN().to(torch.float32)
        def load_data(batch_size):
                train_dataset = torchvision.datasets.FashionMNIST('./data', train=True, transform=T.ToTensor(),download=True)

                test_dataset = torchvision.datasets.FashionMNIST('./data', train=False, 
                                                        transform=T.ToTensor(),
                                                        download=True)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
                return train_loader, test_loader
        learning_rate=0.01
        def train(net, train_loader, criterion, optimizer, num_epochs):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.to(device)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            
            max_memory_allocated_list = []  # Maksimum bellek miktarını depolamak için bir liste oluşturun
            
            for epoch in range(num_epochs):
                net.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Bellek temizleme
                    del images, labels
                
                # Her epoch sonunda GPU belleğinde ayrılan maksimum belleği alın
                max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB cinsinden
                max_memory_allocated_list.append(max_memory_allocated)  # Listeye ekle
                
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = correct / total
                
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
                
                # Bellek temizleme
                del running_loss, correct, total
        
            return max_memory_allocated_list  # Maksimum bellek miktarı listesini döndür

        def test_model(net, test_loader, criterion, device):
            net.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
            accuracy = total_correct / total_samples
            return accuracy

        def save_checkpoint(net, optimizer, path):
            checkpoint = {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, path)

        def load_checkpoint(net, optimizer, path):
            checkpoint = torch.load(path)
            net.load_state_dict(checkpoint["net_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        def setup_logging(log_file):
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            return logger

        def main():
            # Define hyperparameters
            batch_size = 512
            learning_rate = 1e-4
            num_epochs = 5
            model_path = "model.pt"
            log_file = "training.log"
            
            torch.backends.cudnn.benchmark = True

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load data
            train_loader, test_loader = load_data(batch_size)

            # Initialize model, optimizer, criterion
            net = SimpleNN().to(device)
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Train model
            train(net, train_loader, criterion, optimizer, num_epochs)

            # Test model
            test_accuracy = test_model(net, test_loader, criterion, device)
            print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

            # Save model checkpoint
            save_checkpoint(net, optimizer, model_path)

            # Setup logging
            logger = setup_logging(log_file)
            logger.info(f"Test Accuracy: {test_accuracy}")

        main()
    main_SimpleNN()
    pass

