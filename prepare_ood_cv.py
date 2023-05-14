import argparse
import multiprocessing
import os

import gdown
from tqdm import tqdm
import wget

from src.config import load_config, get_abs_path, P3D_CATEGORIES, OODCV_CATEGORIES, MESH_LEN
from src.prepare_data import prepare_pascal3d_sample


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare OOD-CV data')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--workers', type=int, default=6)
    return parser.parse_args()


def download_data(cfg):
    pascal3d_raw_path = get_abs_path(cfg.pascal3d_raw_path)
    pascal3d_occ_raw_path = get_abs_path(cfg.pascal3d_occ_raw_path)
    ood_cv_pose_data_path = get_abs_path(cfg.ood_cv_pose_data_path)

    if os.path.isdir(pascal3d_raw_path):
        print(f"Found Pascal3D+ dataset at {pascal3d_raw_path}")
    else:
        print(f"Downloading Pascal3D+ dataset at {pascal3d_raw_path}")
        wget.download(cfg.pascal3d_raw_url)
        os.system(f"unzip PASCAL3D+_release1.1.zip -d {os.path.dirname(pascal3d_raw_path)}")
        os.system("rm PASCAL3D+_release1.1.zip")

    if os.path.isdir(pascal3d_occ_raw_path):
        print(f"Found OccludedPascal3D+ dataset at {pascal3d_occ_raw_path}")
    else:
        os.makedirs(pascal3d_occ_raw_path, exist_ok=True)
        os.chdir(pascal3d_occ_raw_path)
        print(f"Downloading OccludedPascal3D+ dataset at {pascal3d_occ_raw_path}")
        wget.download(cfg.pascal3d_occ_script_url)
        os.system("chmod +x download_FG.sh")
        os.system("sh download_FG.sh")
        os.chdir("..")

    if os.path.isdir(ood_cv_pose_data_path):
        print(f"Found OOD-CV pose dataset at {ood_cv_pose_data_path}")
    else:
        print(f"Downloading OOD-CV pose dataset at {ood_cv_pose_data_path}")
        gdown.download(cfg.ood_cv_pose_url, output="pose.zip", fuzzy=True)
        os.system(f"unzip pose.zip -d {os.path.dirname(ood_cv_pose_data_path)}")
        os.system("rm pose.zip")


def prepare_pascal3d_data(cfg):
    pascal3d_data_path = get_abs_path(cfg.pascal3d_data_path)
    if os.path.isdir(pascal3d_data_path):
        print(f"Found prepared PASCAL3D+ dataset at {pascal3d_data_path}")
        return

    set_types = ["train", "val"]
    tasks = []
    for _set_type in set_types:
        save_root = os.path.join(pascal3d_data_path, _set_type)
        os.makedirs(save_root, exist_ok=True)
        for occ in getattr(cfg.occ_levels, _set_type):
            for cate in P3D_CATEGORIES:
                tasks.append([cfg, _set_type, occ, cate])

    with multiprocessing.Pool(cfg.args.workers) as pool:
        results = list(tqdm(pool.imap(p3d_worker, tasks), total=len(tasks)))

    total_samples, total_errors = {k: {} for k in set_types}, {
        k: {} for k in set_types
    }
    for (_err, _total, _set, _cate) in results:
        if _cate not in total_samples[_set]:
            total_samples[_set][_cate] = _total
            total_errors[_set][_cate] = _err
        else:
            total_samples[_set][_cate] += _total
            total_errors[_set][_cate] += _err
    for _set_type in set_types:
        for cate in P3D_CATEGORIES:
            print(
                f"Prepared {total_samples[_set_type][cate]} {_set_type} samples for {cate}, "
                f"error rate {total_errors[_set_type][cate]/total_samples[_set_type][cate]*100:.2f}%"
            )


def prepare_oodcv_data(cfg):
    ood_cv_path = get_abs_path(cfg.oodcv_data_path)
    if os.path.isdir(ood_cv_path):
        print(f"Found prepared OOD-CV dataset at {ood_cv_path}")
        return

    set_types = ["train", "val"]
    tasks = []
    for _set_type in set_types:
        save_root = os.path.join(ood_cv_path, _set_type)
        os.makedirs(save_root, exist_ok=True)
        for cate in OODCV_CATEGORIES:
            tasks.append([cfg, _set_type, 0, cate])

    with multiprocessing.Pool(cfg.args.workers) as pool:
        results = list(tqdm(pool.imap(oodcv_worker, tasks), total=len(tasks)))

    total_samples, total_errors = {k: {} for k in set_types}, {
        k: {} for k in set_types
    }
    for (_err, _total, _set, _cate) in results:
        if _cate not in total_samples[_set]:
            total_samples[_set][_cate] = _total
            total_errors[_set][_cate] = _err
        else:
            total_samples[_set][_cate] += _total
            total_errors[_set][_cate] += _err
    for _set_type in set_types:
        for cate in OODCV_CATEGORIES:
            print(
                f"Prepared {total_samples[_set_type][cate]} {_set_type} samples for {cate}, "
                f"error rate {total_errors[_set_type][cate]/total_samples[_set_type][cate]*100:.2f}%"
            )


def p3d_worker(params):
    cfg, set_type, occ, cate = params
    pascal3d_raw_path = get_abs_path(cfg.pascal3d_raw_path)
    pascal3d_occ_raw_path = get_abs_path(cfg.pascal3d_occ_raw_path)
    pascal3d_data_path = get_abs_path(cfg.pascal3d_data_path)
    save_root = os.path.join(pascal3d_data_path, set_type)

    this_size = cfg.image_sizes[cate]
    out_shape = [
        ((this_size[0] - 1) // 32 + 1) * 32,
        ((this_size[1] - 1) // 32 + 1) * 32,
    ]
    out_shape = [int(out_shape[0]), int(out_shape[1])]

    if occ == 0:
        data_name = ""
    else:
        data_name = f"FGL{occ}_BGL{occ}"

    save_image_path = os.path.join(save_root, "images", f"{cate}{data_name}")
    save_annotation_path = os.path.join(save_root, "annotations", f"{cate}{data_name}")
    save_list_path = os.path.join(save_root, "lists", f"{cate}{data_name}")
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_annotation_path, exist_ok=True)
    os.makedirs(save_list_path, exist_ok=True)

    list_dir = os.path.join(pascal3d_raw_path, "Image_sets")
    pkl_dir = os.path.join(pascal3d_raw_path, "Image_subsets")
    anno_dir = os.path.join(pascal3d_raw_path, "Annotations", f"{cate}_imagenet")
    if occ == 0:
        img_dir = os.path.join(pascal3d_raw_path, "Images", f"{cate}_imagenet")
        occ_mask_dir = ""
    else:
        img_dir = os.path.join(pascal3d_occ_raw_path, "images", f"{cate}{data_name}")
        occ_mask_dir = os.path.join(
            pascal3d_occ_raw_path, "annotations", f"{cate}{data_name}"
        )

    list_fname = os.path.join(list_dir, f"{cate}_imagenet_{set_type}.txt")
    with open(list_fname) as fp:
        image_names = fp.readlines()
    image_names = [x.strip() for x in image_names if x != "\n"]

    num_errors = 0
    mesh_name_list = [[] for _ in range(MESH_LEN[cate])]
    for img_name in image_names:
        img_path = os.path.join(img_dir, f"{img_name}.JPEG")
        anno_path = os.path.join(anno_dir, f"{img_name}.mat")

        prepared_sample_names = prepare_pascal3d_sample(
            cate,
            img_name,
            img_path,
            anno_path,
            save_image_path=save_image_path,
            save_annotation_path=save_annotation_path,
            out_shape=out_shape,
            occ_path=None if occ == 0 else os.path.join(occ_mask_dir, f"{img_name}.npz"),
            prepare_mode='first',
            center_and_resize=True)
        if prepared_sample_names is None:
            num_errors += 1
            continue

        for (cad_index, sample_name) in prepared_sample_names:
            mesh_name_list[cad_index - 1].append(sample_name)

    for i, x in enumerate(mesh_name_list):
        with open(
            os.path.join(save_list_path, "mesh%02d" % (i + 1) + ".txt"), "w"
        ) as fl:
            fl.write("\n".join(x))

    return num_errors, len(image_names), set_type, cate


def oodcv_worker(params):
    cfg, set_type, occ, cate = params
    pascal3d_raw_path = get_abs_path(cfg.pascal3d_raw_path)
    ood_cv_pose_data_path = get_abs_path(cfg.ood_cv_pose_data_path)
    ood_cv_path = get_abs_path(cfg.oodcv_data_path)
    save_root = os.path.join(ood_cv_path, set_type)

    this_size = cfg.image_sizes[cate]
    out_shape = [
        ((this_size[0] - 1) // 32 + 1) * 32,
        ((this_size[1] - 1) // 32 + 1) * 32,
    ]
    out_shape = [int(out_shape[0]), int(out_shape[1])]

    if occ == 0:
        data_name = ""
    else:
        data_name = f"FGL{occ}_BGL{occ}"
    save_image_path = os.path.join(save_root, "images", f"{cate}{data_name}")
    save_annotation_path = os.path.join(save_root, "annotations", f"{cate}{data_name}")
    save_list_path = os.path.join(save_root, "lists", f"{cate}{data_name}")
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_annotation_path, exist_ok=True)
    os.makedirs(save_list_path, exist_ok=True)

    if set_type == 'train':
        all_pascal3d_samples = ['.'.join(x.split('.')[:-1]) for x in os.listdir(os.path.join(pascal3d_raw_path, 'Images', f'{cate}_imagenet'))]
        all_pascal3d_samples += ['.'.join(x.split('.')[:-1]) for x in os.listdir(os.path.join(pascal3d_raw_path, 'Images', f'{cate}_pascal'))]
        with open(os.path.join(ood_cv_pose_data_path, 'lists', 'TrainSet.txt'), 'r') as fp:
            all_samples = fp.read().strip().split('\n')
        all_samples = [x for x in all_samples if x in all_pascal3d_samples]
    elif set_type == 'val':
        image_names = []
        nuisances = []

        for n in cfg.nuisances:
            if not os.path.isfile(os.path.join(ood_cv_pose_data_path, 'lists', f'{cate}_{n}.txt')):
                continue
            with open(os.path.join(ood_cv_pose_data_path, 'lists', f'{cate}_{n}.txt'), 'r') as fp:
                fnames = fp.read().strip().split('\n')
            image_names += fnames
            nuisances += [n] * len(fnames)

        with open(os.path.join(ood_cv_pose_data_path, 'lists', f'{cate}_all.txt'), 'r') as fp:
            fnames = fp.read().strip().split('\n')
        fnames = [x for x in fnames if x not in image_names]
        image_names += fnames
        nuisances += ['iid'] * len(fnames)
        all_samples = [x.split(' ') for x in image_names]
    else:
        raise ValueError(f'Unknown set type: {set_type}')

    num_errors = 0
    mesh_name_list = [[] for _ in range(MESH_LEN[cate])]
    for idx, sample in enumerate(all_samples):
        if set_type == 'train':
            img_name = sample
            if img_name.startswith('n'):
                img_path = os.path.join(pascal3d_raw_path, 'Images', f'{cate}_imagenet', f'{img_name}.JPEG')
                anno_path = os.path.join(pascal3d_raw_path, 'Annotations', f'{cate}_imagenet', f'{img_name}.mat')
            else:
                img_path = os.path.join(pascal3d_raw_path, 'Images', f'{cate}_pascal', f'{img_name}.jpg')
                anno_path = os.path.join(pascal3d_raw_path, 'Annotations', f'{cate}_pascal', f'{img_name}.mat')
            obj_ids = None
        elif set_type == 'val':
            img_name, obj_id = sample
            obj_ids = [int(obj_id)]
            img_path = os.path.join(ood_cv_pose_data_path, 'images', cate, f'{img_name}.JPEG')
            anno_path = os.path.join(ood_cv_pose_data_path, 'annotations', cate, f'{img_name}.mat')

        prepared_sample_names = prepare_pascal3d_sample(
            cate,
            img_name,
            img_path,
            anno_path,
            save_image_path=save_image_path,
            save_annotation_path=save_annotation_path,
            out_shape=out_shape,
            occ_path=None,
            prepare_mode='first',
            obj_ids=obj_ids,
            extra_anno=None if set_type == 'train' else {'nuisance': nuisances[idx]},
            center_and_resize=True
        )
        if prepared_sample_names is None:
            num_errors += 1
            continue

        for (cad_index, sample_name) in prepared_sample_names:
            mesh_name_list[cad_index - 1].append(sample_name)

    for i, x in enumerate(mesh_name_list):
        with open(
            os.path.join(save_list_path, "mesh%02d" % (i + 1) + ".txt"), "w"
        ) as fl:
            fl.write("\n".join(x))

    return num_errors, len(all_samples), set_type, cate


def main():
    args = parse_args()
    cfg = load_config(args, load_default_config=False, log_info=False)

    download_data(cfg)
    prepare_pascal3d_data(cfg)
    prepare_oodcv_data(cfg)


if __name__ == '__main__':
    main()
