import os
import matplotlib.pyplot as plt 


def show_image(image, mask, pred_image = None, pred_uq = None):
    if pred_image is None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(), cmap = 'gray')
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(), cmap = 'gray')
        
    elif pred_image is not None and pred_uq is None:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(), cmap = 'gray')
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(), cmap = 'gray')
        ax3.set_title('MODEL OUTPUT')
        ax3.imshow(pred_image.permute(1,2,0).squeeze(), cmap = 'gray')

    elif pred_image is not None and pred_uq is not None:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,5))
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(), cmap = 'gray')
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(), cmap = 'gray')
        ax3.set_title('MODEL OUTPUT')
        ax3.imshow(pred_image.permute(1,2,0).squeeze(), cmap = 'gray')
        ax4.set_title('MODEL UQ')
        ax4.imshow(pred_uq.permute(1,2,0).squeeze(), cmap = 'gray')


def plot_loss(epochs, train_losses, test_losses):
    plt.figure(figsize=(9,9))
    plt.plot(epochs, train_losses, c='r', linewidth=3, label='train loss')
    plt.plot(epochs, test_losses, c='b', linewidth=3, label='test loss')
    plt.xlabel('epoch', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()


def save_image_mask_out(output_path, idx, image, mask, out):
    os.makedirs(output_path, exist_ok=True)
    show_image(image, mask, out.detach().cpu().squeeze(0))
    filename = f'pred_{idx}_.png'
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def save_image_mask_avg_std(output_path, idx, image, mask, avg_out, std_out):
    os.makedirs(output_path, exist_ok=True)
    show_image(image, mask, avg_out.detach().cpu(), std_out.detach().cpu())
    filename = f'pred_{idx}.png'
    plt.savefig(os.path.join(output_path, filename))
    plt.close()