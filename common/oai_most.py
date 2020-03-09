import argparse
import torch
from torch import Tensor
import numpy as np
import os
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from sas7bdat import SAS7BDAT

# Confusion matrix
import re
import itertools
import matplotlib
from textwrap import wrap
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

oai_meta_csv_filename = "oai_meta.csv"
most_meta_csv_filename = "most_meta.csv"

oai_participants_csv_filename = "oai_participants.csv"
most_participants_csv_filename = "most_participants.csv"

oai_most_meta_csv_filename = "oai_most_meta.csv"

oai_most_all_csv_filename = "oai_most_all.csv"
oai_most_img_csv_filename = "oai_most_img_patches.csv"

STD_SZ = (128, 128)

follow_up_dict_most = {0: '00', 1: '15', 2: '30', 3: '60', 5: '84'}
visit_to_month = {"most": ["00", "15", "30", "60", "84"], "oai": ["00", "12", "24", "36", "72", "96"]}


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])


def build_img_klg_meta_oai(oai_src_dir):
    visits = visit_to_month["oai"]
    dataset_name = "oai"
    exam_codes = ['00', '01', '03', '05', '08', '10']
    rows = []
    sides = [None, 'R', 'L']
    for i, visit in enumerate(visits):
        print(f'==> Reading OAI {visit} visit')
        meta = read_sas7bdata_pd(os.path.join(oai_src_dir,
                                              'Semi-Quant Scoring_SAS',
                                              f'kxr_sq_bu{exam_codes[i]}.sas7bdat'))
        # Dropping the data from multiple projects
        meta.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
        meta.fillna(-1, inplace=True)
        for c in meta.columns:
            meta[c.upper()] = meta[c]

        meta['KL'] = meta[f'V{exam_codes[i]}XRKL']

        meta['XROSTL'] = meta[f'V{exam_codes[i]}XROSTL']
        meta['XROSTM'] = meta[f'V{exam_codes[i]}XROSTM']

        meta['XROSFL'] = meta[f'V{exam_codes[i]}XROSFL']
        meta['XROSFM'] = meta[f'V{exam_codes[i]}XROSFM']

        meta['XRJSL'] = meta[f'V{exam_codes[i]}XRJSL']
        meta['XRJSM'] = meta[f'V{exam_codes[i]}XRJSM']

        for index, row in tqdm(meta.iterrows(), total=len(meta.index), desc="Loading OAI meta"):
            _s = int(row['SIDE'])
            if sides[_s] is None:
                continue
            rows.append(
                {'ID': row['ID'], 'Side': sides[_s], 'KL': row['KL'], 'XROSTL': row['XROSTL'], 'XROSTM': row['XROSTM'],
                 'XROSFL': row['XROSFL'], 'XROSFM': row['XROSFM'], 'XRJSL': row['XRJSL'], 'XRJSM': row['XRJSM'],
                 'visit_id': i, 'visit': visit, 'dataset': dataset_name})

    return pd.DataFrame(rows, index=None)


def build_img_klg_meta_most(most_src_dir):
    dataset_name = "most"
    data = read_sas7bdata_pd(os.path.join(most_src_dir, 'mostv01235xray.sas7bdat')).fillna(-1)
    data.set_index('MOSTID', inplace=True)
    rows = []
    # Assumption: Only use visit 0 (baseline)
    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}

    enrolled = {}
    for visit in [0, 1, 2, 3, 5]:
        print(f'==> Reading MOST {visit} visit')
        ds = read_sas7bdata_pd(files_dict[f'mostv{visit}enroll.sas7bdat'])
        if 'V{}PA'.format(visit) in ds:
            ds = ds[ds['V{}PA'.format(visit)] == 1]  # Filtering out the cases when X-rays wern't taken
        id_set = set(ds.MOSTID.values.tolist())
        enrolled[visit] = id_set

    rows = []
    for i, visit in enumerate([0, 1, 2, 3, 5]):
        for ID in tqdm(enrolled[visit], total=len(enrolled[visit]), desc="Loading MOST meta"):
            subj = data.loc[ID]
            kl_l_key = 'V{0}X{1}{2}'.format(visit, 'L', 'KL')
            kl_r_key = 'V{0}X{1}{2}'.format(visit, 'R', 'KL')
            if kl_l_key in subj and kl_r_key in subj:
                KL_bl_l = subj[kl_l_key]
                KL_bl_r = subj[kl_r_key]
                rows.append({'ID': ID, 'Side': 'L', 'KL': KL_bl_l, 'visit_id': i, 'visit': visit_to_month["most"][i],
                             'dataset': dataset_name})
                rows.append({'ID': ID, 'Side': 'R', 'KL': KL_bl_r, 'visit_id': i, 'visit': visit_to_month["most"][i],
                             'dataset': dataset_name})

    return pd.DataFrame(rows, columns=['ID', 'Side', 'KL', 'visit_id', 'visit', 'dataset'])


def get_most_meta(meta_path):
    # SIDES numbering is made according to the OAI notation
    # SIDE=1 - Right
    # SIDE=2 - Left
    print('==> Processing', os.path.join(meta_path, 'mostv01235xray.sas7bdat'))
    most_meta = read_sas7bdata_pd(os.path.join(meta_path, 'mostv01235xray.sas7bdat'))

    most_names_list = pd.read_csv(os.path.join(meta_path, 'MOST_names.csv'), header=None)[0].values.tolist()
    xray_types = pd.DataFrame(
        list(map(lambda x: (x.split('/')[0][:-5], follow_up_dict_most[int(x.split('/')[1][1])], x.split('/')[-2]),
                 most_names_list)), columns=['ID', 'visit', 'TYPE'])

    most_meta_all = []
    for visit_id in [0, 1, 2, 3, 5]:
        for leg in ['L', 'R']:
            features = ['MOSTID', ]
            for compartment in ['L', 'M']:
                for bone in ['F', 'T']:
                    features.append(f"V{visit_id}X{leg}OS{bone}{compartment}"),
                features.append(f"V{visit_id}X{leg}JS{compartment}")
            features.append(f"V{visit_id}X{leg}KL")
            tmp = most_meta.copy()[features]
            trunc_feature_names = list(map(lambda x: 'XR' + x[4:], features[1:]))
            tmp[trunc_feature_names] = tmp[features[1:]]
            tmp.drop(features[1:], axis=1, inplace=True)
            tmp['Side'] = leg  # int(1 if leg == 'R' else 2)
            tmp = tmp[~tmp.isnull().any(1)]
            tmp['visit'] = follow_up_dict_most[visit_id]
            tmp['ID'] = tmp['MOSTID'].copy()
            tmp.drop('MOSTID', axis=1, inplace=True)
            most_meta_all.append(tmp)

    most_meta = pd.concat(most_meta_all)
    most_meta = most_meta[(most_meta[trunc_feature_names] <= 4).all(1)]
    most_meta = pd.merge(xray_types, most_meta)
    most_meta = most_meta[most_meta.TYPE == 'PA10']
    most_meta.drop('TYPE', axis=1, inplace=True)
    return most_meta


def filter_most_by_pa(ds, df_most_ex, pas=['PA10']):
    std_rows = []
    for i, row in df_most_ex.iterrows():
        std_row = dict()
        std_row['ID'] = row['ID_ex'].split('_')[0]
        std_row['visit_id'] = int(row['visit'][1:])
        std_row['PA'] = row['PA']
        std_rows.append(std_row)
    df_most_pa = pd.DataFrame(std_rows)

    ds_most_filtered = pd.merge(ds, df_most_pa, on=['ID', 'visit_id'])
    if isinstance(pas, str):
        ds_most_filtered = ds_most_filtered[ds_most_filtered['PA'] == pas]
    return ds_most_filtered


def build_clinical_oai(oai_src_dir):
    data_enrollees = read_sas7bdata_pd(os.path.join(oai_src_dir, 'enrollees.sas7bdat'))
    data_clinical = read_sas7bdata_pd(os.path.join(oai_src_dir, 'allclinical00.sas7bdat'))

    clinical_data_oai = data_clinical.merge(data_enrollees, on='ID')

    # Age, Sex, BMI
    clinical_data_oai['SEX'] = 2 - clinical_data_oai['P02SEX']
    clinical_data_oai['AGE'] = clinical_data_oai['V00AGE']
    clinical_data_oai['BMI'] = clinical_data_oai['P01BMI']

    clinical_data_oai_left = clinical_data_oai.copy()
    clinical_data_oai_right = clinical_data_oai.copy()

    # Making side-wise metadata
    clinical_data_oai_left['Side'] = 'L'
    clinical_data_oai_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_oai_left['INJ'] = clinical_data_oai_left['P01INJL']
    clinical_data_oai_right['INJ'] = clinical_data_oai_right['P01INJR']

    # Surgery (ever had)
    clinical_data_oai_left['SURG'] = clinical_data_oai_left['P01KSURGL']
    clinical_data_oai_right['SURG'] = clinical_data_oai_right['P01KSURGR']

    # Total WOMAC score
    clinical_data_oai_left['WOMAC'] = clinical_data_oai_left['V00WOMTSL']
    clinical_data_oai_right['WOMAC'] = clinical_data_oai_right['V00WOMTSR']

    clinical_data_oai = pd.concat((clinical_data_oai_left, clinical_data_oai_right))
    clinical_data_oai.ID = clinical_data_oai.ID.values.astype(str)
    return clinical_data_oai[['ID', 'Side', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC']]


def build_clinical_most(most_src_dir):
    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}
    clinical_data_most = read_sas7bdata_pd(files_dict['mostv0enroll.sas7bdat'])
    clinical_data_most['ID'] = clinical_data_most.MOSTID
    clinical_data_most['BMI'] = clinical_data_most['V0BMI']

    clinical_data_most_left = clinical_data_most.copy()
    clinical_data_most_right = clinical_data_most.copy()

    # Making side-wise metadata
    clinical_data_most_left['Side'] = 'L'
    clinical_data_most_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_most_left['INJ'] = clinical_data_most_left['V0LAL']
    clinical_data_most_right['INJ'] = clinical_data_most_right['V0LAR']

    # Surgery (ever had)
    clinical_data_most_left['SURG'] = clinical_data_most_left['V0SURGL']
    clinical_data_most_right['SURG'] = clinical_data_most_right['V0SURGR']

    # Total WOMAC score
    clinical_data_most_left['WOMAC'] = clinical_data_most_left['V0WOTOTL']
    clinical_data_most_right['WOMAC'] = clinical_data_most_right['V0WOTOTR']

    clinical_data_most = pd.concat((clinical_data_most_left, clinical_data_most_right))

    return clinical_data_most[['ID', 'Side', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC']]


def load_most_metadata(root, save_dir, force_reload=False):
    most_meta_fullname = os.path.join(save_dir, most_meta_csv_filename)
    most_participants_fullname = os.path.join(save_dir, most_participants_csv_filename)
    most_all_fullname = os.path.join(save_dir, oai_most_all_csv_filename)

    requires_update = False

    if os.path.isfile(most_meta_fullname) and not force_reload:
        most_meta = pd.read_csv(most_meta_fullname, sep='|')
    else:
        most_meta = build_img_klg_meta_most(os.path.join(root, 'most_meta/'))
        most_meta_strict = get_most_meta(os.path.join(root, 'most_meta/'))

        most_meta = pd.merge(most_meta, most_meta_strict, on=('ID', 'Side', 'visit'), how='inner')
        most_meta.to_csv(most_meta_fullname, index=None, sep='|')
        requires_update = True

    if os.path.isfile(most_participants_fullname) and not force_reload:
        most_ppl = pd.read_csv(most_participants_fullname, sep='|')
    else:
        most_ppl = build_clinical_most(os.path.join(root, 'most_meta/'))
        most_ppl.to_csv(most_participants_fullname, index=None, sep='|')
        requires_update = True

    master_dict = {"oai": dict(), "most": dict()}
    master_dict["most"]["meta"] = most_meta
    master_dict["most"]["ppl"] = most_ppl

    master_dict["most"]["n_dup"] = dict()

    master_dict["most"]["n_dup"]["meta"] = len(most_meta[most_meta.duplicated(keep=False)].index)
    master_dict["most"]["n_dup"]["ppl"] = len(most_ppl[most_ppl.duplicated(keep=False)].index)

    for ds in ["most"]:
        for s in ["meta", "ppl"]:
            if master_dict[ds]["n_dup"][s] > 0:
                print(master_dict[ds][s])
                raise ValueError(
                    "There are {} duplicated rows in {} {} dataframe".format(master_dict[ds]["n_dup"][s], ds.upper(),
                                                                             s.upper()))

    master_dict["most"]["all"] = pd.merge(master_dict["most"]["meta"],
                                          master_dict["most"]["ppl"], how="left",
                                          left_on=["ID", "Side"], right_on=["ID", "Side"]).fillna(-1)

    return master_dict


def load_oai_most_metadata(root, save_dir, force_reload=False):
    oai_meta_fullname = os.path.join(save_dir, oai_meta_csv_filename)
    oai_participants_fullname = os.path.join(save_dir, oai_participants_csv_filename)
    most_meta_fullname = os.path.join(save_dir, most_meta_csv_filename)
    most_participants_fullname = os.path.join(save_dir, most_participants_csv_filename)
    oai_most_meta_fullname = os.path.join(save_dir, oai_most_meta_csv_filename)
    oai_most_all_fullname = os.path.join(save_dir, oai_most_all_csv_filename)

    requires_update = False

    if os.path.isfile(oai_meta_fullname) and not force_reload:
        oai_meta = pd.read_csv(oai_meta_fullname, sep='|')
        oai_meta["ID"] = oai_meta["ID"].values.astype(str)
    else:
        oai_meta = build_img_klg_meta_oai(os.path.join(root, 'X-Ray_Image_Assessments_SAS/'))
        oai_meta.to_csv(oai_meta_fullname, index=None, sep='|')
        requires_update = True

    if os.path.isfile(oai_participants_fullname) and not force_reload:
        oai_ppl = pd.read_csv(oai_participants_fullname, sep='|')
    else:
        oai_ppl = build_clinical_oai(os.path.join(root, 'X-Ray_Image_Assessments_SAS/'))
        oai_ppl.to_csv(oai_participants_fullname, index=None, sep='|')
        requires_update = True

    if os.path.isfile(most_meta_fullname) and not force_reload:
        most_meta = pd.read_csv(most_meta_fullname, sep='|')
    else:
        most_meta = build_img_klg_meta_most(os.path.join(root, 'most_meta/'))
        most_meta_strict = get_most_meta(os.path.join(root, 'most_meta/'))
        most_meta = pd.merge(most_meta, most_meta_strict, on=('ID', 'Side', 'visit'), how='inner')
        most_meta.to_csv(most_meta_fullname, index=None, sep='|')
        requires_update = True

    if os.path.isfile(most_participants_fullname) and not force_reload:
        most_ppl = pd.read_csv(most_participants_fullname, sep='|')
    else:
        most_ppl = build_clinical_most(os.path.join(root, 'most_meta/'))
        most_ppl.to_csv(most_participants_fullname, index=None, sep='|')
        requires_update = True

    master_dict = {"oai": dict(), "most": dict()}
    master_dict["oai"]["meta"] = oai_meta
    master_dict["oai"]["ppl"] = oai_ppl
    master_dict["most"]["meta"] = most_meta
    master_dict["most"]["ppl"] = most_ppl

    master_dict["oai"]["n_dup"] = dict()
    master_dict["most"]["n_dup"] = dict()
    master_dict["oai"]["n_dup"]["meta"] = len(oai_meta[oai_meta.duplicated(keep=False)].index)
    master_dict["oai"]["n_dup"]["ppl"] = len(oai_ppl[oai_ppl.duplicated(keep=False)].index)
    master_dict["most"]["n_dup"]["meta"] = len(most_meta[most_meta.duplicated(keep=False)].index)
    master_dict["most"]["n_dup"]["ppl"] = len(most_ppl[most_ppl.duplicated(keep=False)].index)

    for ds in ["oai", "most"]:
        for s in ["meta", "ppl"]:
            if master_dict[ds]["n_dup"][s] > 0:
                print(master_dict[ds][s])
                raise ValueError(
                    "There are {} duplicated rows in {} {} dataframe".format(master_dict[ds]["n_dup"][s], ds.upper(),
                                                                             s.upper()))

    master_dict["oai_most"] = dict()
    master_dict["oai_most"]["meta"] = pd.concat([master_dict["oai"]["meta"],
                                                 master_dict["most"]["meta"]], ignore_index=True)

    master_dict["oai"]["all"] = pd.merge(master_dict["oai"]["meta"],
                                         master_dict["oai"]["ppl"], how="left",
                                         left_on=["ID", "Side"], right_on=["ID", "Side"]).fillna(-1)
    master_dict["most"]["all"] = pd.merge(master_dict["most"]["meta"],
                                          master_dict["most"]["ppl"], how="left",
                                          left_on=["ID", "Side"], right_on=["ID", "Side"]).fillna(-1)

    master_dict["oai_most"]["all"] = pd.concat([master_dict["oai"]["all"],
                                                master_dict["most"]["all"]], ignore_index=True)

    if requires_update:
        master_dict["oai_most"]["meta"].to_csv(oai_most_meta_fullname, index=None, sep='|')
        master_dict["oai_most"]["all"].to_csv(oai_most_all_fullname, index=None, sep='|')

    return master_dict


def crop_2_rois_oai_most(img, ps=128, debug=False):
    """
    Generates pair of images 128x128 from the knee joint.
    ps shows how big area should be mapped into that region.
    """
    if len(img.shape) > 2:
        if img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
        else:
            raise ValueError('Input image ({}) must have 1 channel, but got {}'.format(img.shape, img.shape[0]))
    elif len(img.shape) < 2:
        raise ValueError(
            'Input image ({}) must have 2 dims or 1-channel 3 dims, but got {}'.format(img.shape, img.shape))
    elif img.shape[0] < 300 or img.shape[1] < 300:
        raise ValueError('Input image shape ({}) must be at least 300, but got {}'.format(img.shape, img.shape))

    s = img.shape[0]

    s1, s2, pad = calc_roi_bboxes(s, ps)

    # pad = int(np.floor(s / 3))
    # l = img[pad:pad+ps, 0:ps]
    # m = img[pad:pad+ps, s-ps:s]

    l = img[s1[0]:s1[2], s1[1]:s1[3]]
    m = img[s2[0]:s2[2], s2[1]:s2[3]]

    # DEBUG
    if debug:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (0, pad), (ps, pad + ps), color=(255, 255, 0), thickness=4)
        cv2.rectangle(img, (s - ps, pad), (s, pad + ps), color=(255, 255, 0), thickness=4)
        return l, m, img
    else:
        return l, m, None


def calc_roi_bboxes(s, ps):
    pad = int(np.floor(s / 3))

    s1 = (pad, 0, pad + ps, ps)
    s2 = (pad, s - ps, pad + ps, s)
    return s1, s2, pad


def standardize_img(img, std_actual_shape=(130, 130), target_shape=(300, 300),
                    original_actual_shape=(140, 140), original_img_shape=(700, 700)):
    spacing_per_pixel = (1.0 * original_actual_shape[0] / original_img_shape[0],
                         1.0 * original_actual_shape[1] / original_img_shape[1])
    std_img_shape = (int(std_actual_shape[0] / spacing_per_pixel[0]), int(std_actual_shape[1] / spacing_per_pixel[1]))
    cropped_img = center_crop(img, std_img_shape)
    cropped_img = cv2.resize(cropped_img, dsize=target_shape, interpolation=cv2.INTER_AREA)
    return cropped_img


def overlay_heatmap(heatmap, mask, whole_img, box, blend_w=0.7, std_actual_shape=(110, 110), target_shape=(300, 300),
                    original_actual_shape=(140, 140), original_img_shape=(700, 700), draw_crops=False,
                    crop_center=False):
    spacing_per_pixel = (1.0 * original_actual_shape[0] / original_img_shape[0],
                         1.0 * original_actual_shape[1] / original_img_shape[1])
    std_img_shape = (int(std_actual_shape[0] / spacing_per_pixel[0]), int(std_actual_shape[1] / spacing_per_pixel[1]))

    s = std_img_shape[0] / target_shape[0]

    heatmap = cv2.resize(heatmap, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)

    box = list(box)

    for i in range(len(box)):
        box[i] *= s

    c_y = int(round((whole_img.shape[0] - std_img_shape[0]) / 2))
    c_x = int(round((whole_img.shape[1] - std_img_shape[1]) / 2))

    box[0] += c_y
    box[1] += c_x
    box[2] += c_y
    box[3] += c_x

    if whole_img.shape[2] == 1:
        whole_img = cv2.cvtColor(whole_img, cv2.COLOR_GRAY2BGR)

    box = [int(round(box[i])) for i in range(len(box))]

    # print(f'box shape = ({box[2]-box[0]}, {box[3]-box[1]})')

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    if len(heatmap.shape) == 2:
        heatmap = np.expand_dims(heatmap, axis=-1)

    whole_img = np.float32(whole_img) / 255
    blend_whole_img = blend_heatmap(whole_img, heatmap, mask, blend_w, box)

    if draw_crops:
        cv2.rectangle(whole_img, (box[1], box[0]), (box[3], box[2]), (0, 123, 247), 4)

    blend_whole_img = np.uint8(255 * blend_whole_img)

    return blend_whole_img, heatmap, mask, box


def blend_heatmap(img, heatmap, mask, w=0.8, box=None):
    if box is None:
        img = 1 * (1 - mask ** w) * img + (mask ** w) * heatmap
    elif (isinstance(box, tuple) or isinstance(box, list)) and len(box) == 4:
        img[box[0]:box[2], box[1]:box[3], :] = 1 * (1 - mask ** w) * img[box[0]:box[2], box[1]:box[3], :] + (
                mask ** w) * heatmap
    else:
        raise ValueError(f'Invalid input box with type of {type(box)}')

    return img


def center_crop(img, crop_sz):
    c_y = int(round((img.shape[0] - crop_sz[0]) / 2))
    c_x = int(round((img.shape[1] - crop_sz[1]) / 2))
    cropped_img = img[c_y:c_y + crop_sz[0], c_x:c_x + crop_sz[1]]
    return cropped_img


def load_oai_most_datasets(root, img_dir, save_meta_dir, saved_patch_dir, output_filename, force_reload=False,
                           force_rewrite=False, extract_sides=True):
    if not os.path.exists(save_meta_dir):
        os.mkdir(save_meta_dir)

    if not os.path.exists(saved_patch_dir):
        os.mkdir(saved_patch_dir)

    if force_rewrite or not os.path.exists(os.path.join(save_meta_dir, output_filename)):
        print('Loading OAI MOST metadata...')
        df_meta = load_oai_most_metadata(root=root, save_dir=save_meta_dir, force_reload=force_reload)
        print('OK!')
        df = df_meta["oai_most"]["all"]
        fullnames = []
        for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Processing images"):
            fname = row["ID"] + "_" + visit_to_month[row["dataset"]][row["visit_id"]] + "_" + row["Side"] + ".png"
            img_fullname = os.path.join(img_dir, fname)
            basename = os.path.splitext(fname)[0]

            img1_fullname = os.path.join(saved_patch_dir, "{}_patch1.png".format(basename))
            img2_fullname = os.path.join(saved_patch_dir, "{}_patch2.png".format(basename))

            img_cropped_fullname = os.path.join(saved_patch_dir, "{}_cropped.png".format(basename))

            fullnames_dict = row.to_dict()
            fullnames_dict['Filename'] = None
            fullnames_dict['Patch1_name'] = None
            fullnames_dict['Patch2_name'] = None

            fullnames_dict['ID'] = row['ID']
            fullnames_dict['Side'] = row['Side']
            if os.path.exists(img_fullname) and row["KL"] > -1 and row["KL"] < 5:
                if (not os.path.isfile(img_cropped_fullname) and not extract_sides) or \
                        ((not os.path.isfile(img1_fullname) or not os.path.isfile(img2_fullname)) and extract_sides):
                    img = cv2.imread(img_fullname, cv2.IMREAD_GRAYSCALE)

                    img = standardize_img(img, std_actual_shape=(110, 110), target_shape=(300, 300))
                    if extract_sides:
                        img1, img2, _ = crop_2_rois_oai_most(img)
                        cv2.imwrite(img1_fullname, img1)
                        cv2.imwrite(img2_fullname, img2)
                    else:
                        cv2.imwrite(img_cropped_fullname, img)

                fullnames_dict['Filename'] = fname
                if extract_sides:
                    fullnames_dict['Patch1_name'] = os.path.basename(img1_fullname)
                    fullnames_dict['Patch2_name'] = os.path.basename(img2_fullname)
                else:
                    fullnames_dict['ROI_name'] = os.path.basename(img_cropped_fullname)
                fullnames.append(fullnames_dict)

        df_all = pd.DataFrame(fullnames, index=None)
        df_all['ID'] = df_all['ID'].astype(str)
        df_all.to_csv(os.path.join(save_meta_dir, output_filename), index=False, sep='|')
    else:
        df_all = pd.read_csv(os.path.join(save_meta_dir, output_filename), sep='|')
    return df_all


def load_most_dataset(root, img_dir, save_meta_dir, saved_patch_dir, output_filename, force_reload=False,
                      force_rewrite=False):
    if not os.path.exists(save_meta_dir):
        os.mkdir(save_meta_dir)

    if not os.path.exists(saved_patch_dir):
        os.mkdir(saved_patch_dir)

    if force_rewrite or not os.path.exists(os.path.join(save_meta_dir, output_filename)):
        df_meta = load_most_metadata(root=root, save_dir=save_meta_dir, force_reload=force_reload)
        df = df_meta["most"]["all"]
        fullnames = []
        for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Processing images"):
            fname = row["ID"] + "_" + visit_to_month[row["dataset"]][row["visit_id"]] + "_" + row["Side"] + ".png"
            img_fullname = os.path.join(img_dir, fname)
            basename = os.path.splitext(fname)[0]
            img1_fullname = os.path.join(saved_patch_dir, "{}_patch1.png".format(basename))
            img2_fullname = os.path.join(saved_patch_dir, "{}_patch2.png".format(basename))
            fullnames_dict = row.to_dict()
            fullnames_dict['Filename'] = None
            fullnames_dict['Patch1_name'] = None
            fullnames_dict['Patch2_name'] = None

            fullnames_dict['ID'] = row['ID']
            fullnames_dict['Side'] = row['Side']
            if not os.path.exists(img_fullname):
                print('Not found file {}'.format(img_fullname))
            elif row["KL"] < 0 and row["KL"] > 4:
                print('KL {} is out of range'.format(row["KL"]))
            else:
                if not os.path.isfile(img1_fullname) or not os.path.isfile(img2_fullname):
                    img = cv2.imread(img_fullname, cv2.IMREAD_GRAYSCALE)

                    img = standardize_img(img, std_actual_shape=(110, 110), target_shape=(300, 300))
                    img1, img2, _ = crop_2_rois_oai_most(img)
                    cv2.imwrite(img1_fullname, img1)
                    cv2.imwrite(img2_fullname, img2)

                fullnames_dict['Filename'] = fname
                fullnames_dict['Patch1_name'] = os.path.basename(img1_fullname)
                fullnames_dict['Patch2_name'] = os.path.basename(img2_fullname)
                fullnames.append(fullnames_dict)

        df_all = pd.DataFrame(fullnames, index=None)
        print('Save {} lines into {}'.format(len(df_all), os.path.join(save_meta_dir, output_filename)))
        df_all.to_csv(os.path.join(save_meta_dir, output_filename), index=False, sep='|')
    else:
        df_all = pd.read_csv(os.path.join(save_meta_dir, output_filename), sep='|')
    return df_all
