import torch.nn as nn


class M3(nn.Module):
    def __init__(self) -> None:
        super(M3, self).__init__()
        self.conv_list = nn.ModuleList([self.bn2d(1, 32)])
        for i in range(32, 161, 16):
            self.conv_list.append(self.bn2d(i, i+16))
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11264, 10), nn.BatchNorm1d(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        return self.FC(x)

    def bn2d(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


if __name__ == "__main__":
    # test
    import torch
    import matplotlib.pyplot as plt
    import torchvision.transforms as tf
    import torchvision as tv
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    test_loader = DataLoader(
        tv.datasets.MNIST("data", transform=tf.ToTensor(), train=False),
        batch_size=25,
        shuffle=True,
        num_workers=10
    )  # num_workers 根据机子实际情况设置，一般<=CPU数

    net = M3().to(device)
    net.load_state_dict(
        torch.load("model_params_0.996399998664856.pth")
    )

    with torch.no_grad():
        rows, cols = 5, 5
        num_pics = 5

        for imgs, targets in test_loader:
            preds = torch.argmax(net.forward(
                imgs.to(device=device)).detach().cpu(), dim=1)

            fig, axs = plt.subplots(rows, cols)
            plt.rcParams.update({'font.size': 7})

            for r in range(rows):
                for c in range(cols):
                    axs[r, c].imshow(imgs[r*5+c].squeeze(0))
                    axs[r, c].axis("off")
                    axs[r, c].set_title(f"pred {int(preds[r*5+c])}")

            plt.tight_layout()
            plt.savefig(f"verify{num_pics}.png")
            plt.close()

            num_pics -= 1
            if num_pics == 0:
                break
