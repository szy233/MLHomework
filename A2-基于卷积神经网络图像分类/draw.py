import matplotlib.pyplot as plt


with open('eval-acc.txt', 'r') as file:
    content = file.read()
    eval_acc_value = [float(num) for num in content.split(',')]
    file.close()

with open('train-loss.txt', 'r') as file:
    content = file.read()
    train_loss_value = [float(num) for num in content.split(',')]
    file.close()

with open('eval-loss.txt', 'r') as file:
    content = file.read()
    eval_loss_value = [float(num) for num in content.split(',')]
    file.close()


plt.plot(range(1, len(train_loss_value) + 1), train_loss_value, marker='o', markersize=3, label='train', color='blue')
plt.plot(range(1, len(train_loss_value) + 1), eval_loss_value, marker='^', markersize=3, label='eval', color='red')
plt.text(0.05, 0.95, '⚪ train', color='blue', transform=plt.gca().transAxes)
plt.text(0.05, 0.9, '▲ eval', color='red', transform=plt.gca().transAxes)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.clf()


plt.plot(range(1, len(train_loss_value) + 1), eval_acc_value, marker='^', markersize=3, label='eval', color='y')
plt.text(0.05, 0.95, '▲ eval', color='y', transform=plt.gca().transAxes)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('accuracy.png')
plt.clf()



