import torch
import main

def eval(separation):
    model = main.SomethingSimple()
    model.load_state_dict(torch.load("model.params"))

    data_loader = main.fashion_mnist_dataloader(separation)

    total, corrects = 0., 0.
    for images, targets in data_loader:
        predicitons = torch.argmax(model(images), dim=1)
        correct_predictions = (predicitons == targets).sum()

        total += float(len(targets))
        corrects += float(correct_predictions)

    print(f"Accuracy: {corrects/total}, on {separation}")

if __name__=="__main__":
    eval(main.Separation.TRAIN)
    eval(main.Separation.TEST)
