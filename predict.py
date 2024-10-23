from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# 加载模型
model = load_model('my_model.h5')

# CIFAR-10 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 加载并预处理图片
img = image.load_img('/root/.vscode-server/data/CNN-image_classfication/ship.png', target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 进行预测
predictions = model.predict(img_array)

# 输出预测类别
predicted_class = np.argmax(predictions)
print("Predicted class:", class_names[predicted_class])
