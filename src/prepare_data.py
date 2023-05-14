import os

import BboxTools as bbt
import cv2
import math
import numpy as np
from PIL import Image
import scipy.io as sio

mesh_para_names = [
    "azimuth", "elevation", "theta", "distance",
    "focal", "principal", "viewport", "cad_index",
    "bbox"]


def get_obj_ids(record, cate=None):
    check_keys = [
        "azimuth", "elevation", "distance", "focal",
        "theta", "principal", "viewport", "height",
        "width", "cad_index", "bbox"]
    ids = []
    for i in range(record["objects"][0].shape[0]):
        try:
            get_anno(record, *check_keys, idx=i)
        except IndexError:
            continue
        except ValueError:
            continue

        if cate is not None and get_anno(record, "category", idx=i) != cate:
            continue

        ids.append(i)
    return ids


def get_anno(record, *args, idx=0):
    out = []
    objects = record["objects"][0]
    viewpoint = record["objects"][0][idx]["viewpoint"][0][0]
    for key_ in args:
        if key_ == "category" or key_ == "class":
            out.append(str(objects[idx]["class"][0]))
        elif key_ == "height":
            out.append(record["imgsize"][0][1])
        elif key_ == "width":
            out.append(record["imgsize"][0][0])
        elif key_ == "bbox":
            out.append(objects[idx]["bbox"][0])
        elif key_ == "cad_index":
            out.append(objects[idx]["cad_index"].item())
        elif key_ == "principal":
            px = viewpoint["px"].item()
            py = viewpoint["py"].item()
            out.append(np.array([px, py]))
        elif key_ in ["theta", "azimuth", "elevation"]:
            if type(viewpoint[key_].item()) == tuple:
                tmp = viewpoint[key_].item()[0]
            else:
                tmp = viewpoint[key_].item()
            out.append(tmp * math.pi / 180)
        elif key_ == "distance":
            if type(viewpoint["distance"].item()) == tuple:
                distance = viewpoint["distance"].item()[0]
            else:
                distance = viewpoint["distance"].item()
            out.append(distance)
        else:
            out.append(viewpoint[key_].item())

    if len(out) == 1:
        return out[0]

    return tuple(out)


def prepare_pascal3d_sample(
    cate,
    img_name,
    img_path,
    anno_path,
    save_image_path,
    save_annotation_path,
    out_shape,
    occ_path=None,
    prepare_mode="first",
    obj_ids=None,
    center_and_resize=True,
    extra_anno=None
):
    if not os.path.isfile(img_path):
        print(f'Image file {img_path} not found')
        return None
    if not os.path.isfile(anno_path):
        print(f'Annotation file {anno_path} not found')
        return None

    mat_contents = sio.loadmat(anno_path)
    record = mat_contents["record"][0][0]
    if occ_path is not None:
        occ_mask = np.load(occ_path, allow_pickle=True)["occluder_mask"].astype(np.uint8)
    else:
        occ_mask = None

    if obj_ids is None:
        obj_ids = get_obj_ids(record, cate=cate)
        if len(obj_ids) == 0:
            return None
        if prepare_mode == "first":
            if obj_ids[0] != 0:
                return []
            else:
                obj_ids = [0]

    img = np.array(Image.open(img_path))
    _h, _w = img.shape[0], img.shape[1]

    save_image_names = []
    for obj_id in obj_ids:
        bbox = get_anno(record, "bbox", idx=obj_id)
        box = bbt.from_numpy(bbox, sorts=("x0", "y0", "x1", "y1"))

        if get_anno(record, "distance", idx=obj_id) <= 0:
            continue

        if center_and_resize:
            target_distances = [5.0]
            dist = get_anno(record, "distance", idx=obj_id)
            all_resize_rates = [float(dist / x) for x in target_distances]
        else:
            all_resize_rates = [
                min(out_shape[0] / img.shape[0], out_shape[1] / img.shape[1])]

        for rr_idx, resize_rate in enumerate(all_resize_rates):
            if resize_rate < 0.001:
                resize_rate = min(out_shape[0] / box.shape[0], out_shape[1] / box.shape[1])
            try:
                box_ori = bbt.from_numpy(bbox, sorts=("x0", "y0", "x1", "y1"))
                box = bbt.from_numpy(bbox, sorts=("x0", "y0", "x1", "y1")) * resize_rate

                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = np.array(img)
                box_ori = box_ori.set_boundary(img.shape[0:2])

                dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
                img = cv2.resize(img, dsize=dsize)

                if center_and_resize:
                    center = (
                        get_anno(record, "principal", idx=obj_id)[::-1] * resize_rate
                    ).astype(int)
                else:
                    center = np.array([img.shape[0]//2, img.shape[1]//2]).astype(np.int32)
                box1 = bbt.box_by_shape(out_shape, center)
                if (
                    out_shape[0] // 2 - center[0] > 0
                    or out_shape[1] // 2 - center[1] > 0
                    or out_shape[0] // 2 + center[0] - img.shape[0] > 0
                    or out_shape[1] // 2 + center[1] - img.shape[1] > 0
                ):
                    padding = (
                        (
                            max(out_shape[0] // 2 - center[0], 0),
                            max(out_shape[0] // 2 + center[0] - img.shape[0], 0),
                        ),
                        (
                            max(out_shape[1] // 2 - center[1], 0),
                            max(out_shape[1] // 2 + center[1] - img.shape[1], 0),
                        ),
                        (0, 0),
                    )

                    img = np.pad(img, padding, mode="constant")

                    if occ_mask is not None:
                        occ_mask = np.pad(occ_mask, (padding[0], padding[1]), mode='constant')

                    box = box.shift([padding[0][0], padding[1][0]])
                    box1 = box1.shift([padding[0][0], padding[1][0]])
                else:
                    padding = ((0, 0), (0, 0), (0, 0))

                box_in_cropped = box.copy()
                box = box1.set_boundary(img.shape[0:2])
                box_in_cropped = box.box_in_box(box_in_cropped)

                bbox = box.bbox
                img_cropped = img[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :]
                if occ_mask is not None:
                    occ_mask = occ_mask[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

            except KeyboardInterrupt:
                continue

            curr_img_name = f"{img_name}_{obj_id:02d}"

            save_parameters = dict(
                name=img_name,
                box=box.numpy(),
                box_ori=box_ori.numpy(),
                box_obj=box_in_cropped.numpy(),
                # cropped_kp_list=cropped_kp_list,
                # visible=states_list,
                occ_mask=occ_mask,
            )
            save_parameters = {
                **save_parameters,
                **{
                    k: v
                    for k, v in zip(
                        mesh_para_names, get_anno(record, *mesh_para_names, idx=obj_id)
                    )
                },
            }
            save_parameters["height"] = _h
            save_parameters["width"] = _w
            save_parameters["resize_rate"] = resize_rate
            save_parameters["padding_params"] = np.array(
                [
                    padding[0][0],
                    padding[0][1],
                    padding[1][0],
                    padding[1][1],
                    padding[2][0],
                    padding[2][1],
                ]
            )

            if extra_anno is not None:
                for k in extra_anno:
                    save_parameters[k] = extra_anno[k]

            np.savez(
                os.path.join(save_annotation_path, curr_img_name), **save_parameters
            )
            Image.fromarray(img_cropped).save(
                os.path.join(save_image_path, curr_img_name + ".JPEG")
            )
            save_image_names.append(
                (get_anno(record, "cad_index", idx=obj_id), curr_img_name)
            )

    return save_image_names
