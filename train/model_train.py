import os


# Training function
def train_triple_u_net(model, train_loader, val_loader=None, num_epochs=100, save_path='checkpoints'):
    os.makedirs(save_path, exist_ok=True)

    # you need to change the path to your tensorboard log files
    writer = SummaryWriter('tf-logs')

    
    pre_d_loss = 0
    pre_g_loss = 10000
    
    for epoch in range(num_epochs):
        model.generator.train()
        model.discriminator.train()

        epoch_losses = {
            'loss_d': 0.0,
            'loss_g_gan': 0.0,
            'loss_g_l1': 0.0,
            'loss_g': 0.0,
            'loss_G_perc': 0.0
        }

        for i, (m_img, bg_img, people_img, target_img) in enumerate(train_loader):
            losses = model.train_step(m_img, bg_img, people_img, target_img)

            # Update running losses
            for k, v in losses.items():
                epoch_losses[k] += v

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                      f"D_loss: {losses['loss_d']:.4f}, G_loss: {losses['loss_g']:.4f}")
    
        # Print epoch stats
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
            writer.add_scalar(k, epoch_losses[k], epoch)
        
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"D_loss: {epoch_losses['loss_d']:.4f}, "
              f"G_GAN_loss: {epoch_losses['loss_g_gan']:.4f}, "
              f"G_L1_loss: {epoch_losses['loss_g_l1']:.4f}, "
              f"G_perc_loss: {epoch_losses['loss_G_perc']:.4f}, "
              f"G_total_loss: {epoch_losses['loss_g']:.4f}, "
        )

        # Save model checkpoint
        if epoch_losses['loss_d'] >= pre_d_loss and epoch_losses['loss_g'] <= pre_g_loss:
            model.save_models(os.path.join(save_path, f'model_best_epoch.pth'))
            pre_d_loss = epoch_losses['loss_d']
            pre_g_loss = epoch_losses['loss_g']
            print(f'Save best! epoch [{epoch+1}]!')
            
        if (epoch + 1) % 10 == 0:
            model.save_models(os.path.join(save_path, f'model_epoch_{epoch + 1}.pth'))

    writer.close()