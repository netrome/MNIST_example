import torch
import main

def eval():
    model = main.SomethingSimple()
    model.load_state_dict(torch.load("model.params"))

    data_loader = main.fashion_mnist_dataloader(main.Separation.TEST)

    total, corrects = 0., 0.
    for images, targets in data_loader:
        predicitons = torch.argmax(model(images), dim=1)
        correct_predictions = (predicitons == targets).sum()

        total += float(len(targets))
        corrects += float(correct_predictions)

    print(f"Accuracy: {corrects/total}")

if __name__=="__main__":
    eval()
