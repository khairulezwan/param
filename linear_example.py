import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

labels = ["dog", "cat", "panda"]
np.random.seed(1)

W = np.random.randn(3,3072)
b = np.random.randn(3)

orig = cv2.imread("dog.png")
image = cv2.resize(orig, (32,32)).flatten()

scores = W.dot(image) + b

for (label, score) in zip(labels,scores):
    print("[INFO] {}: {:.2f}".format(label,score))

cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
            (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)

plt.axis("off")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.show()
