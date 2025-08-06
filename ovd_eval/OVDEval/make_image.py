import matplotlib
# 强制使用无界面 Agg 后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# IOU 阈值
iou = [0.1, 0.3, 0.5, 0.7]

# 将下面的列表替换成你实际测得的准确率
accuracy_my       = [0.1, 0.0656, 0.0612, 0.06]  # 示例值，请替换
accuracy_simple   = [0.1, 0.0645, 0.060, 0.0588]  # 示例值，请替换
accuracy_original = [0.1, 0.0612, 0.0392, 0.0351]  # 示例值，请替换

plt.figure()
plt.plot(iou, accuracy_my,       marker='o', label='my')
plt.plot(iou, accuracy_simple,   marker='s', label='simple')
plt.plot(iou, accuracy_original, marker='^', label='original')

plt.xlabel('IOU')
plt.ylabel('Accuracy')
plt.xticks(iou)            # 确保横坐标只显示这些阈值
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('logo_iou_accuracy.png')
plt.show()
