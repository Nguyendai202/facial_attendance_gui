
import copy
import time
import argparse
import cv2 as cv
from yunet.yunet_onnx import YuNetONNX

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='model/face_detection_yunet_120x160.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="160,120",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.6,
        help='Conf confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.3,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--keep_topk',
        type=int,
        default=750,
    )

    args = parser.parse_args()

    return args


def main():
    # truyền đối số
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    topk = args.topk
    keep_topk = args.keep_topk
    yunet = YuNetONNX(
        model_path=model_path,
        input_shape=input_shape,
        conf_th=score_th,
        nms_th=nms_th,
        topk=topk,
        keep_topk=keep_topk,
    )
    return yunet


def draw_debug(
    image,
    elapsed_time,
    score_th,
    input_shape,
    bboxes,
    landmarks,
    scores,
):
    image_width, image_height = image.shape[1], image.shape[0]
    for bbox, landmark, score in zip(bboxes, landmarks, scores):
        if score_th > score:
            continue
        # bouding box
        x1 = int(image_width * (bbox[0] / input_shape[0]))
        y1 = int(image_height * (bbox[1] / input_shape[1]))
        x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
        y2 = int(image_height * (bbox[3] / input_shape[1])) + y1
    return x1, y1, x2, y2


if __name__ == '__main__':
    get_args()
    main()
    x1, y1, x2, y2 = draw_debug()
