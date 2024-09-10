# CV2 显示图片
# 使用 make_grid 将张量组合成一个网格
grid_img = vutils.make_grid(img, nrow=3, normalize=True)
grid_img_np = grid_img.permute(1, 2, 0).numpy()
# 如果原始张量是 RGB 格式，需要转换为 BGR 格式
grid_img_np = cv2.cvtColor(grid_img_np, cv2.COLOR_RGB2BGR)
cv2.imshow('Grid Image', grid_img_np)
cv2.waitKey(1)
cv2.destroyAllWindows()

# test时代码
output_img = transforms.ToPILImage()(output.cpu().squeeze(0))
output_img.save(predict_path + img_name + '.png')

img_tensor_4dim = torch.unsqueeze(img_tensor_3dim, dim=0)
out = net(img_tensor_4dim)