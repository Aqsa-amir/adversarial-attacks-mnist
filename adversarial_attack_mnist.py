# Adversarial attacks (FGSM, PGD) and adversarial training on MNIST
# Works as a normal script or in Google Colab (each "# %%" block is a cell).
# In Colab: Runtime > Change runtime type > T4 GPU (optional, CPU also works).

# %% 1. Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print("Using device:", device)

# %% 2. Data (pixels stay in [0, 1] so epsilon is easy to interpret)
tf = transforms.ToTensor()
train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)


# %% 3. Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


# %% 4. Attacks
def fgsm(model, x, y, eps):
    """One-step attack: move every pixel by eps in the direction that raises the loss."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x_adv + eps * grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha=None, steps=10):
    """Iterative attack: many small FGSM steps, each projected back into the eps-ball."""
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1).detach()
    return x_adv


# %% 5. Training and evaluation helpers
def train(model, epochs=3, adv_eps=None):
    """Standard training if adv_eps is None, otherwise adversarial training (PGD)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if adv_eps is not None:
                x_adv = pgd(model, x, y, adv_eps, steps=5)
                x = torch.cat([x, x_adv])  # train on clean + adversarial
                y = torch.cat([y, y])
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {ep + 1}/{epochs}  loss {total / len(train_loader):.4f}")
    return model


def accuracy(model, attack=None, eps=0.0):
    model.eval()
    correct = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        if attack is not None and eps > 0:
            x = attack(model, x, y, eps)
        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
    return 100 * correct / len(test_ds)


# %% 6. Train the normal (undefended) model
print("Training standard model...")
std_model = train(CNN().to(device), epochs=3)
print(f"Clean test accuracy: {accuracy(std_model):.2f}%")

# %% 7a. Attack the standard model with FGSM
eps_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
std_fgsm = [accuracy(std_model, fgsm, e) for e in eps_list]
for e, a in zip(eps_list, std_fgsm):
    print(f"eps={e:.2f}  FGSM acc={a:.2f}%")

# %% 7b. Attack the standard model with PGD
std_pgd = [accuracy(std_model, pgd, e) for e in eps_list]
for e, b in zip(eps_list, std_pgd):
    print(f"eps={e:.2f}  PGD acc={b:.2f}%")

# %% 8a. Train the adversarially robust model, Defense: adversarial training
print("Training adversarially robust model (slower)...")
adv_model = train(CNN().to(device), epochs=3, adv_eps=0.2)
print(f"Clean test accuracy: {accuracy(adv_model):.2f}%")

# %% 8b. FGSM on the robust model
adv_fgsm = [accuracy(adv_model, fgsm, e) for e in eps_list]
for e, a in zip(eps_list, adv_fgsm):
    print(f"eps={e:.2f}  FGSM acc={a:.2f}%")

# %% 8c. PGD on the robust model
adv_pgd = [accuracy(adv_model, pgd, e) for e in eps_list]
for e, b in zip(eps_list, adv_pgd):
    print(f"eps={e:.2f}  PGD acc={b:.2f}%")

# %% 9. Plot accuracy vs. attack strength
plt.figure(figsize=(7, 4.5))
plt.plot(eps_list, std_fgsm, "o--", label="Standard model, FGSM")
plt.plot(eps_list, std_pgd, "o-", label="Standard model, PGD")
plt.plot(eps_list, adv_fgsm, "s--", label="Adv. trained, FGSM")
plt.plot(eps_list, adv_pgd, "s-", label="Adv. trained, PGD")
plt.xlabel("Attack strength (epsilon)")
plt.ylabel("Test accuracy (%)")
plt.title("MNIST: accuracy under adversarial attack")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_vs_epsilon.png", dpi=150)
plt.show()

# %% 10. Show example adversarial images
std_model.eval()
x, y = next(iter(test_loader))
x, y = x[:6].to(device), y[:6].to(device)
x_adv = pgd(std_model, x, y, eps=0.2)
with torch.no_grad():
    pred_clean = std_model(x).argmax(1)
    pred_adv = std_model(x_adv).argmax(1)

fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
for i in range(6):
    axes[0, i].imshow(x[i, 0].cpu(), cmap="gray")
    axes[0, i].set_title(f"clean: {pred_clean[i].item()}")
    axes[1, i].imshow(x_adv[i, 0].cpu(), cmap="gray")
    axes[1, i].set_title(f"attacked: {pred_adv[i].item()}")
    axes[0, i].axis("off")
    axes[1, i].axis("off")
plt.tight_layout()
plt.savefig("adversarial_examples.png", dpi=150)
plt.show()
