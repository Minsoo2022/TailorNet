import os
import numpy as np
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import torch

from psbody.mesh import Mesh

from models.tailornet_model import get_best_runner as get_tn_runner
from models.smpl4garment import SMPL4Garment
from utils.rotation import normalize_y_rotation
from visualization.blender_renderer import visualize_garment_body

from dataset.canonical_pose_dataset import get_style, get_shape
from visualization.vis_utils import get_specific_pose, get_specific_style_old_tshirt
from visualization.vis_utils import get_specific_shape, get_amass_sequence_thetas
from utils.interpenetration import remove_interpenetration_fast
from utils.smpl_paths import get_hres

# Set output path where inference results will be stored
BCNET_SMPL_PATH = "../TailorNet_dataset/ROOT/smpl"
BCNET_MODEL_NUM = "4"
OUT_PATH = "./results"


def get_single_frame_inputs(garment_class, gender):
    """Prepare some individual frame inputs."""
    smpl = pkl.load(open(os.path.join(BCNET_SMPL_PATH, 'bcnet_model_{}.pkl'.format(BCNET_MODEL_NUM)), 'rb'))
    theta, beta = smpl['pose'][0], smpl['shape'][0]
    theta[0, 1] *= -1
    theta = R.from_matrix(theta).as_rotvec().reshape(-1)

    thetas = [theta for _ in range(5)]
    betas = [beta for _ in range(5)]
    # old t-shirt style parameters are centered around [1.5, 0.5, 1.5, 0.0]
    # whereas all other garments styles are centered around [0, 0, 0, 0]
    if garment_class == 'old-t-shirt':
        gammas = [
            get_specific_style_old_tshirt('mean'),
            get_specific_style_old_tshirt('big'),
            get_specific_style_old_tshirt('small'),
            get_specific_style_old_tshirt('shortsleeve'),
            get_specific_style_old_tshirt('big_shortsleeve'),
        ]
    else:
        gammas = [
            np.array([0, 0, 0, 0]),
            np.array([-1, 0, 0, 0]),
            np.array([1, 0, 0, 0]),
            np.array([0, -1, 0, 0]),
            np.array([0, 1, 0, 0]),
        ]
    return thetas, betas, gammas


def get_sequence_inputs(garment_class, gender):
    """Prepare sequence inputs."""
    beta = get_specific_shape('somethin')
    if garment_class == 'old-t-shirt':
        gamma = get_specific_style_old_tshirt('big_longsleeve')
    else:
        gamma = get_style('000', gender=gender, garment_class=garment_class)

    # downsample sequence frames by 2
    thetas = get_amass_sequence_thetas('05_02')[::2]

    betas = np.tile(beta[None, :], [thetas.shape[0], 1])
    gammas = np.tile(gamma[None, :], [thetas.shape[0], 1])
    return thetas, betas, gammas


def run_tailornet():
    gender = 'male'
    garment_class = 'short-pant'
    garment_class_pairs = {
        'shirt': [],
        't-shirt': [],
        'pant': ['shirt', 't-shirt'],
        'short-pant': ['shirt', 't-shirt'],
        'skirt': ['shirt', 't-shirt']
    }
    thetas, betas, gammas = get_single_frame_inputs(garment_class, gender)
    # # uncomment the line below to run inference on sequence data
    # thetas, betas, gammas = get_sequence_inputs(garment_class, gender)

    # load model
    tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)
    # from trainer.base_trainer import get_best_runner
    # tn_runner = get_best_runner("/BS/cpatel/work/data/learn_anim/tn_baseline/{}_{}/".format(garment_class, gender))
    smpl = SMPL4Garment(gender=gender)

    # make out directory if doesn't exist
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    # run inference
    for i, (theta, beta, gamma) in enumerate(zip(thetas, betas, gammas)):
        print(i, len(thetas))
        # normalize y-rotation to make it front facing
        theta_normalized = normalize_y_rotation(theta)
        with torch.no_grad():
            pred_verts_d = tn_runner.forward(
                thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()

        # get garment from predicted displacements
        body, pred_gar = smpl.run(beta=beta, theta=theta, garment_class=garment_class, garment_d=pred_verts_d)
        pred_gar = remove_interpenetration_fast(pred_gar, body)

        hv, hf, mapping = get_hres(pred_gar.v, pred_gar.f)
        pred_gar_hres = Mesh(hv, hf)

        for garment_class_pair in garment_class_pairs[garment_class]:
            gar_pair_hres = Mesh(filename=os.path.join(OUT_PATH, "bcnet_model_{}_gar_hres_{}_{:04d}.obj".format(BCNET_MODEL_NUM, garment_class_pair, i)))
            # print(len(body.v), body.f.min(), body.f.max())
            # print(len(pred_gar_hres.v), pred_gar_hres.f.min(), pred_gar_hres.f.max())
            # print(np.vstack((body.v, pred_gar_hres.v)).shape, np.min(pred_gar_hres.f + len(body.v)), np.max(pred_gar_hres.f + len(body.v)))
            # gar_pair_hres = remove_interpenetration_fast(gar_pair_hres, pred_gar_hres)
            for _ in range(3):
                gar_pair_hres = remove_interpenetration_fast(gar_pair_hres, Mesh(np.vstack((body.v, pred_gar_hres.v)), np.vstack((body.f, pred_gar_hres.f + len(body.v)))))
            gar_pair_hres.write_obj(os.path.join(OUT_PATH, "bcnet_model_{}_gar_hres_{}_inter_{}_{:04d}.obj".format(BCNET_MODEL_NUM, garment_class_pair, garment_class, i)))

        # save body and predicted garment
        body.write_obj(os.path.join(OUT_PATH, "bcnet_model_{}.obj".format(BCNET_MODEL_NUM)))
        pred_gar.write_obj(os.path.join(OUT_PATH, "bcnet_model_{}_gar_{}_{:04d}.obj".format(BCNET_MODEL_NUM, garment_class, i)))
        pred_gar_hres.write_obj(os.path.join(OUT_PATH, "bcnet_model_{}_gar_hres_{}_{:04d}.obj".format(BCNET_MODEL_NUM, garment_class, i)))


def render_images():
    """Render garment and body using blender."""
    i = 0
    while True:
        body_path = os.path.join(OUT_PATH, "body_{:04d}.ply".format(i))
        if not os.path.exists(body_path):
            break
        body = Mesh(filename=body_path)
        pred_gar = Mesh(filename=os.path.join(OUT_PATH, "pred_gar_{:04d}.ply".format(i)))

        visualize_garment_body(
            pred_gar, body, os.path.join(OUT_PATH, "img_{:04d}.png".format(i)), garment_class='t-shirt', side='front')
        i += 1

    # Concate frames of sequence data using this command
    # ffmpeg -r 10 -i img_%04d.png -vcodec libx264 -crf 10  -pix_fmt yuv420p check.mp4
    # Make GIF
    # convert -delay 200 -loop 0 -dispose 2 *.png check.gif
    # convert check.gif -resize 512x512 check_small.gif


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1 or sys.argv[1] == 'inference':
        run_tailornet()
    elif sys.argv[1] == 'render':
        render_images()
    else:
        raise AttributeError
