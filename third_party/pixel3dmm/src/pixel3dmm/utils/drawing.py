from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import io
import cv2


def get_img_from_fig(fig, w=256, h=256, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.resize(cv2.imdecode(img_arr, 1), (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_points(image, pts, pts2):

    fig = plt.figure(frameon=False)
    #fig.set_size_inches(w,h)
    plt.axis('off')
    #ax = plt.Axes(fig, [0., 0., 1., 1.])

    #fig.add_axes(ax)

    #ax.imshow(your_image, aspect='auto')
    plt.imshow(image)
    #plt.plot(640, 570, "og", markersize=10)  # og:shorthand for green circle
    plt.scatter(pts[:, 0], pts[:, 1], marker="o", color="red", s=1)
    plt.scatter(pts2[:, 0], pts2[:, 1], marker="o", color="green", s=1)
    plt.plot(np.stack([pts[:, 0], pts2[:, 0]], axis=0), np.stack([pts[:, 1], pts2[:, 1]], axis=0), '-b', linewidth=0.2)
    #plt.plot()
    #plt.show()
    fig.canvas.draw()

    #data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    data = get_img_from_fig(fig)
    plt.close(fig)
    return data


if __name__ == '__main__':
    image = np.ones([1024, 1024, 3], dtype=np.uint8) * 255
    pts = np.array([[330, 620], [950, 620], [692, 450], [587, 450]])
    pts2 = np.array([[330, 620], [950, 620], [692, 450], [587, 450]])
    pts2 = pts2 * 0.75


    data = plot_points(image, pts, pts2)
    print(data.shape)
    Image.fromarray(data).show()


