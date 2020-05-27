import mxnet as mx
from gluoncv import model_zoo, data, utils


class YoloV3:
    def __init__(self):
        """Yolo V3 pretrained model
           VOC dataset: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
        """
        self.net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

    @staticmethod
    def preprocess(image):
        """Load and transform the image
        """
        return data.transforms.presets.yolo.transform_test(mx.ndarray.array(image), short=512)

    def get_object_labels(self, class_ids, scores, score_th=0.5):
        """Filter detections by score and obtain the corresponding label for each class
        """
        class_ids = class_ids[0].asnumpy().reshape(1, -1)
        scores = scores[0].asnumpy().reshape(1, -1)
        selected_class_ids = class_ids[scores > score_th]
        return [self.net.classes[int(class_id)] for class_id in selected_class_ids]

    def detect_objects(self, image, show=False):
        """Object detection in an image
        """
        x, img = self.preprocess(image)
        class_ids, scores, bboxs = self.net(x)

        if show:
            from matplotlib import pyplot as plt
            utils.viz.plot_bbox(img, bboxs[0], scores[0], class_ids[0], class_names=self.net.classes)
            plt.show()

        return class_ids, scores, bboxs

    def __call__(self, *args, **kwargs):
        """1. Detect objects
           2. Get object labels. For this task we are only interested in what elements are in the image, we don't need more information
        """
        image = kwargs.get('image')
        class_ids, scores, _ = self.detect_objects(image, show=False)
        labels = self.get_object_labels(class_ids, scores)
        return labels


if __name__ == '__main__':
    import argparse
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument('--path_image', required=True, help='path to image')
    args = ap.parse_args()
    image = Image.open(args.path_image)

    yolov3 = YoloV3()
    yolov3(image=image)





